#include "Storage_mode.h"
#include "Config.h"

namespace {
  int32_t kCompressedMode = 1;
  int32_t kDataDistributionEnabled = 2;
}  // anonymous namespace

//bool IsCompressedStorageEnabled() {
//  return GLOBAL_FLAG(StorageMode) & kCompressedMode;
//}

bool IsDataDistributionEnabled() {
  return true;
  //return GLOBAL_FLAG(StorageMode) & kDataDistributionEnabled;
}