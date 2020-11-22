#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_STORAGE_MODE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_STORAGE_MODE_H_

/// 0 --> row mode
/// 1 --> column mode
/// 2 --> row mode & data distribution enabled
/// 3 --> column mode & data distribution enabled
//
/// Default is 2, the same behavior like before.

bool IsCompressedStorageEnabled();
bool IsDataDistributionEnabled();

#endif