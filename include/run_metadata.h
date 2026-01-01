#ifndef RUN_METADATA_H
#define RUN_METADATA_H

#include <filesystem>
#include <string>

#include "backend.h"
#include "run_config.h"

std::filesystem::path MetadataSidecarPath(const std::filesystem::path& output_path);

std::string BuildRunMetadataJson(const RunConfig& run_config,
                                 BackendKind requested_backend,
                                 BackendKind selected_backend,
                                 const std::string& backend_note,
                                 const std::string& output_path,
                                 bool time_series,
                                 int frame_index,
                                 double frame_time);

bool WriteRunMetadataSidecar(const std::filesystem::path& output_path,
                             const std::string& metadata_json,
                             std::string* error);
bool WriteRunMetadataSidecar(const std::filesystem::path& output_path,
                             const RunConfig& run_config,
                             BackendKind requested_backend,
                             BackendKind selected_backend,
                             const std::string& backend_note,
                             bool time_series,
                             int frame_index,
                             double frame_time,
                             std::string* error);
bool ReadRunMetadataSidecar(const std::filesystem::path& output_path,
                            std::string* metadata_json,
                            std::string* error);

#endif  // RUN_METADATA_H
