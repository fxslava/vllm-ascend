# TurboQuant source refactor verification

`SOURCE_PACKET.json` identifies the source-only review candidate in both
repositories. The packet pins the matching source inventories and the
baseline commits. `TOKEN_PRESERVATION.json` compares C++ tokens with the
baseline while excluding comments, whitespace and moved include directives.
It records the intentional launch, address and ABI declaration changes.
The E8 reader's six macro continuation changes preserve all 5,220
compared tokens after line splicing.

The pinned standalone and plugin inventories contain 34 byte-identical C++
sources and one byte-identical E8 analyzer metadata file. The production
kernel and wrapper sources contain no comments. Copyright and license text
is retained in the source-tree `NOTICE` files. The raw BF16 E8 encoder was
outside this refactor. Prepared E8 cache packets have a 4,352-byte ABI,
while the reader's internal K group stride is 1,056 bytes. Native decode
takes caller-owned workspace. E8 dequantization has a zero-scratch contract
and the reader uses caller-owned outputs and witnesses.

`VALIDATION.json` indexes the verification commands and hashes. CANN
9.2.1 BiSheng passed the plugin's 10 and standalone's 14 syntax-only checks
using each source file's includes. GCC syntax checks covered the host
contract gates and the five Torch binding and ABI translation units.
The sync and build-selection pytest suite passed 14 tests.
The mocked editable install checks the CMake-selected native libraries,
including the sibling TurboQuant DSOs. Strict editable installs still have
the inherited limitation that the generated ACLNN vendor tree is outside
the setuptools link map. The wheel and source-checkout ACLNN copy paths
are preserved.

The native Python oracle regenerated all 57 recorded binary fixtures
byte-for-byte under Linux with Torch 2.10.0+cpu and NumPy 1.26.4. The E8
Python semantic port checked all 65,536 labels over 16 scales, including
8,388,608 coordinate-scale pairs and 16,384 K payload addresses. A separate
Windows run with Torch 2.14.0+cpu matched all but two stage-1 fixture files.
The files differed by 16 bytes and 8 bytes respectively under that stack.

The original E8 reader's 11 layout and reuse annotations are preserved
verbatim in a shared metadata file. The analyzer wrapper reinstates them
for comparison against the untouched independently verified source. A
second comparison removes the same annotations from the independent
baseline and runs the comment-free candidate directly. Both comparisons
have zero added diagnostics under their matching inputs. The unannotated
reader reports 23 fatal, 11 warning and 3 informational diagnostics, and
the annotated reader reports 33 fatal, 3 warning and 3 informational
diagnostics. Native decode retains 2,896 fatal and 123 warning diagnostics.
The analyzer does not inline the E8 component's two `Dequant` calls from
its entry, so the component's zero diagnostic count has limited call
coverage. The fresh run found zero added diagnostic fingerprints across
all 10 paired scopes, with no parser errors. See
`ANALYZER_SUMMARY.json` for the complete comparison.
The final analyzer driver requires a fresh, complete result for every
selected source. A targeted rerun checked annotated and unannotated E8
reader and NZ scatter results. A separate failed-tool test verified that
existing reports cannot satisfy a fresh invocation.

This verification ran no linker, CAModel, NPU workload or performance
measurement. The historical Wave-v11, Frozen03 and E8 results remain in
[the original champion report](../../wave_champions_20261125/REPORT.md).
