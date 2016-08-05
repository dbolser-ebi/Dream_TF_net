#ifndef BOUND_H
#define BOUND_H

#endif // BOUND_H
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <gsl/gsl_statistics.h>
#include <set>

using namespace boost::filesystem;
using namespace boost::iostreams;
using namespace boost::algorithm;

std::vector<std::string> get_bound_locations(path entry);
void process_directory(std::string dirpath);
