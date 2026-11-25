#include "e8_decode.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
int main()try {
 using namespace e8_decode;
 static_assert(kUbStorageBytes==158464&&kUbStorageBytes<=192u*1024u);
 static_assert(Packed(0,0)==2688&&Packed(0,1)==39424&&Packed(1,0)==75168&&Packed(1,1)==111904);
 static_assert(kUbScores==144960&&kUbContext==145984);
 static_assert(KeyPackedIndex(31,255)*4+4<=kKeyPackedAllocation-kPad);
 for(unsigned owner=0;owner<2;++owner)for(unsigned row=0;row<32;++row)for(unsigned c=0;c<256;++c) {
  const unsigned src=KeyPackedIndex(row,c)*4;
  const unsigned segment=src/1056,within=src%1056;
  if(within>=1024)throw std::runtime_error("payload in gap");
  const unsigned copied=owner*1024+segment*2048+within;
  const unsigned dense=(c/8)*64*8*4+(owner*32+row)*8*4+(c%8)*4;
  if(copied!=dense)throw std::runtime_error("K UB->L1 byte order mismatch");
 }
 for(unsigned coord=0;coord<8;++coord) {
  std::array<unsigned,16> banks{};
  for(unsigned lane=0;lane<32;++lane)++banks[((lane*1056+coord*4)/32)%16];
  for(auto n:banks)if(n!=2)throw std::runtime_error("bank dispersion address-model mismatch");
 }
 std::puts("PASS K payload mapping 16384/16384 floats across owners; 32 lanes cover16 address-bank residues twice; no bank-stall claim; V unchanged");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
