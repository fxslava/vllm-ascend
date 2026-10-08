/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// moe_json: a minimal, dependency-free JSON reader for checkpoint config
// files (RFC 8259 subset, DOM style).
//
// WHY HAND-WRITTEN: this tree is deliberately self-contained (see
// CMakeLists.txt) -- it builds inside a bare CANN container with no package
// fetch, and the only consumer is ModelConfig, which reads a handful of
// scalars off a well-formed HuggingFace config.json. A single-header vendor
// of nlohmann/json is ~25k lines for the same job.
//
// Scope: full value grammar (null / bool / number / string with escapes and
// surrogate pairs / arrays / nested objects), line:col errors, a depth bound
// against hostile nesting. NOT provided: serialization, comments (JSON has
// none), duplicate-key detection (last assignment wins, std::map semantics),
// big-precision integers (doubles hold every field this runner reads).

#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ascend_moe {
namespace moe_json {

class ParseError : public std::runtime_error {
 public:
  ParseError(const std::string& message, size_t line, size_t column)
      : std::runtime_error(message + " at line " + std::to_string(line) + ", column " + std::to_string(column)) {}
};

class Value {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_bool() const { return type_ == Type::kBool; }
  bool is_number() const { return type_ == Type::kNumber; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_object() const { return type_ == Type::kObject; }

  bool as_bool() const { Require(Type::kBool, "boolean"); return bool_; }
  double as_number() const { Require(Type::kNumber, "number"); return number_; }
  const std::string& as_string() const { Require(Type::kString, "string"); return string_; }

  // Integral read of a JSON number. Config files write dimensions without a
  // decimal point, but "43.0" is still accepted: it is exactly representable.
  int64_t as_int() const {
    Require(Type::kNumber, "number");
    if (number_ != static_cast<double>(static_cast<int64_t>(number_))) {
      throw std::runtime_error("JSON number is not an integer: " + std::to_string(number_));
    }
    return static_cast<int64_t>(number_);
  }

  // Object access. `find` returns nullptr when the key is absent -- the
  // reader's way of distinguishing "field missing" from "field wrong".
  const Value* find(const char* key) const {
    Require(Type::kObject, "object");
    for (const auto& entry : object_) {
      if (entry.first == key) {
        return &entry.second;
      }
    }
    return nullptr;
  }

  bool contains(const char* key) const { return find(key) != nullptr; }

  // Array access.
  size_t size() const { Require(Type::kArray, "array"); return array_.size(); }
  const Value& operator[](size_t index) const {
    Require(Type::kArray, "array");
    return array_.at(index);
  }

  static Value MakeNull() { return Value(Type::kNull); }
  static Value MakeBool(bool value) {
    Value value_out(Type::kBool);
    value_out.bool_ = value;
    return value_out;
  }
  static Value MakeNumber(double value) {
    Value value_out(Type::kNumber);
    value_out.number_ = value;
    return value_out;
  }
  static Value MakeString(std::string value) {
    Value value_out(Type::kString);
    value_out.string_ = std::move(value);
    return value_out;
  }
  static Value MakeArray() { return Value(Type::kArray); }
  static Value MakeObject() { return Value(Type::kObject); }

  void Append(Value value) { array_.push_back(std::move(value)); }
  void SetMember(std::string key, Value value) {
    for (auto& entry : object_) {
      if (entry.first == key) {
        entry.second = std::move(value);  // last assignment wins
        return;
      }
    }
    object_.emplace_back(std::move(key), std::move(value));
  }

 private:
  explicit Value(Type type) : type_(type) {}

  void Require(Type expected, const char* name) const {
    if (type_ != expected) {
      throw std::runtime_error(std::string("JSON value is a ") + Name(type_) + ", expected " + name);
    }
  }

  static const char* Name(Type type) {
    switch (type) {
      case Type::kNull: return "null";
      case Type::kBool: return "boolean";
      case Type::kNumber: return "number";
      case Type::kString: return "string";
      case Type::kArray: return "array";
      case Type::kObject: return "object";
    }
    return "?";
  }

  Type type_ = Type::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Value> array_;
  std::vector<std::pair<std::string, Value>> object_;
};

// Parses `text` completely; any trailing non-whitespace is an error.
Value Parse(const std::string& text);

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

namespace detail {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value ParseDocument() {
    Value value = ParseValue(0);
    SkipWhitespace();
    if (position_ != text_.size()) {
      Fail("trailing characters after the JSON document");
    }
    return value;
  }

 private:
  // Deep nesting is either a bug or an attack; either way it must not take
  // the host stack down. Real config files nest four levels.
  static constexpr int kMaxDepth = 64;

  void Fail(const std::string& message) const { throw ParseError(message, line_, column_); }

  char Peek() const {
    if (position_ >= text_.size()) {
      Fail("unexpected end of input");
    }
    return text_[position_];
  }

  char Next() {
    const char consumed = Peek();
    ++position_;
    if (consumed == '\n') {
      ++line_;
      column_ = 1;
    } else {
      ++column_;
    }
    return consumed;
  }

  void SkipWhitespace() {
    while (position_ < text_.size()) {
      const char at = text_[position_];
      if (at != ' ' && at != '\t' && at != '\r' && at != '\n') {
        return;
      }
      Next();
    }
  }

  void Expect(char expected) {
    if (Next() != expected) {
      Fail(std::string("expected '") + expected + "'");
    }
  }

  bool ConsumeIf(char expected) {
    if (position_ < text_.size() && text_[position_] == expected) {
      Next();
      return true;
    }
    return false;
  }

  Value ParseValue(int depth) {
    if (depth > kMaxDepth) {
      Fail("nesting deeper than " + std::to_string(kMaxDepth));
    }
    SkipWhitespace();
    switch (Peek()) {
      case 'n': return ParseLiteral("null", Value::MakeNull());
      case 't': return ParseLiteral("true", Value::MakeBool(true));
      case 'f': return ParseLiteral("false", Value::MakeBool(false));
      case '"': return Value::MakeString(ParseString());
      case '[': return ParseArray(depth);
      case '{': return ParseObject(depth);
      default: return ParseNumber();
    }
  }

  Value ParseLiteral(const char* literal, Value value) {
    for (const char* at = literal; *at != '\0'; ++at) {
      if (Next() != *at) {
        Fail(std::string("invalid literal, expected '") + literal + "'");
      }
    }
    return value;
  }

  Value ParseArray(int depth) {
    Expect('[');
    Value array = Value::MakeArray();
    SkipWhitespace();
    if (ConsumeIf(']')) {
      return array;
    }
    while (true) {
      array.Append(ParseValue(depth + 1));
      SkipWhitespace();
      if (ConsumeIf(',')) {
        continue;
      }
      Expect(']');
      return array;
    }
  }

  Value ParseObject(int depth) {
    Expect('{');
    Value object = Value::MakeObject();
    SkipWhitespace();
    if (ConsumeIf('}')) {
      return object;
    }
    while (true) {
      SkipWhitespace();
      if (Peek() != '"') {
        Fail("expected a string key");
      }
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      object.SetMember(std::move(key), ParseValue(depth + 1));
      SkipWhitespace();
      if (ConsumeIf(',')) {
        continue;
      }
      Expect('}');
      return object;
    }
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (true) {
      const char at = Next();
      if (at == '"') {
        return out;
      }
      if (static_cast<unsigned char>(at) < 0x20) {
        Fail("unescaped control character in string");
      }
      if (at != '\\') {
        out.push_back(at);
        continue;
      }
      switch (Next()) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': AppendUnicodeEscape(&out); break;
        default: Fail("invalid string escape");
      }
    }
  }

  // One \uXXXX escape: a Basic Multilingual Plane code point, or half of a
  // surrogate pair. Encoded back out as UTF-8.
  void AppendUnicodeEscape(std::string* out) {
    uint32_t code = ParseHex4();
    if (code >= 0xD800 && code <= 0xDBFF) {
      if (!(ConsumeIf('\\') && ConsumeIf('u'))) {
        Fail("unpaired high surrogate");
      }
      const uint32_t low = ParseHex4();
      if (low < 0xDC00 || low > 0xDFFF) {
        Fail("unpaired high surrogate");
      }
      code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
    } else if (code >= 0xDC00 && code <= 0xDFFF) {
      Fail("unpaired low surrogate");
    }
    if (code < 0x80) {
      out->push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (code >> 6)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (code >> 12)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (code >> 18)));
      out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  uint32_t ParseHex4() {
    uint32_t value = 0;
    for (int digit = 0; digit < 4; ++digit) {
      const char at = Next();
      value <<= 4;
      if (at >= '0' && at <= '9') {
        value |= static_cast<uint32_t>(at - '0');
      } else if (at >= 'a' && at <= 'f') {
        value |= static_cast<uint32_t>(at - 'a' + 10);
      } else if (at >= 'A' && at <= 'F') {
        value |= static_cast<uint32_t>(at - 'A' + 10);
      } else {
        Fail("invalid \\u escape");
      }
    }
    return value;
  }

  Value ParseNumber() {
    const size_t start = position_;
    if (ConsumeIf('-')) {
    }
    while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
      Next();
    }
    if (ConsumeIf('.')) {
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
        Next();
      }
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      Next();
      if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) {
        Next();
      }
      while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
        Next();
      }
    }
    if (position_ == start) {
      Fail("expected a value");
    }
    const std::string token = text_.substr(start, position_ - start);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == nullptr || *end != '\0') {
      Fail("malformed number '" + token + "'");
    }
    return Value::MakeNumber(value);
  }

  const std::string& text_;
  size_t position_ = 0;
  size_t line_ = 1;
  size_t column_ = 1;
};

}  // namespace detail

inline Value Parse(const std::string& text) {
  detail::Parser parser(text);
  return parser.ParseDocument();
}

}  // namespace moe_json
}  // namespace ascend_moe
