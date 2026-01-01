#ifndef PROGRESS_H
#define PROGRESS_H

#include <functional>
#include <string>

using ProgressCallback = std::function<void(const std::string& phase, double progress)>;

#endif
