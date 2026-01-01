#ifndef VALUE_SAMPLING_H
#define VALUE_SAMPLING_H

#include <vector>

#include "../GlViewer.h"

double SampleFieldValue(size_t idx, GlViewer::FieldType field_type,
                        const std::vector<double>& grid,
                        const struct DerivedFields* derived,
                        bool has_derived);

#endif

