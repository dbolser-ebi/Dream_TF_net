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


using namespace boost::filesystem;
using namespace boost::iostreams;
using namespace boost::algorithm;

#define SZ(v) (int)v.size()

template <class T>
double bound_distance(std::vector<T>& a, std::vector<T>& b)
{
    double tot_bound = 0;
    double similar = 0;
    for(int i=0; i<SZ(a); i++){
        if(a[i] || b[i])
            tot_bound++;
        if(a[i] && b[i])
            similar++;
    }
    return similar/(tot_bound+1);
}

template <class T>
void print_distances(std::vector<std::vector<T>>& tracks, std::vector<std::string>& tissue_names)
{
    int num_tissues = SZ(tissue_names);
    for(int i=0; i<num_tissues-1; i++)
    {
        for(int j=i+1; j<num_tissues; j++)
        {
            std::cout << tissue_names[i] << " " << tissue_names[j]
                         <<": "
                         <<"bound "
                         << bound_distance(tracks[i], tracks[j])
                         <<" corr "
                         << gsl_stats_correlation((double*)tracks[i].data(), 1, (double*)tracks[j].data(), 1, SZ(tracks[i]))
                         << std::endl;
        }
    }
}

void read_file(path entry)
{
    // Decompress gzip file in memory and redirect to ss
    std::stringstream ss;
    ifstream file(entry, std::ios_base::in | std::ios_base::binary);
    filtering_streambuf<input> in;
    in.push(gzip_decompressor());
    in.push(file);
    boost::iostreams::copy(in, ss);
    std::string line;

    // read header file
    std::vector<std::string> tissue_names;
    std::getline(ss, line);
    std::stringstream ssh(line);
    int num_tissues = -3;
    std::string token;
    while(ssh >> token)
    {
        if(num_tissues >= 0)
        {
            tissue_names.push_back(token);
        }
        num_tissues++;
    }
    std::vector<std::vector<double>> tracks(num_tissues, std::vector<double>());


    // read file line by line
    while(std::getline(ss, line))
    {
        std::stringstream ssl(line);
        for(int i=0; i<3; i++){
            std::string token;
            ssl >> token;
        }
        for(int i=0; i<num_tissues; i++)
        {
            char state;
            ssl >> state;
            if (state == 'U')
                tracks[i].push_back(0);
            else if (state == 'A')
                tracks[i].push_back(0);
            else if (state == 'B')
                tracks[i].push_back(1);
        }
    }
    std::cout << "Reading TF " << entry << "completed" << std::endl;

    print_distances(tracks, tissue_names);
}

void read_directory()
{
    std::string dir_path = "../../data/chipseq_labels/";
    boost::filesystem::path p(dir_path);

    for(directory_iterator it(dir_path); it!=directory_iterator(); ++it)
    {
        if (ends_with(it->path().string(), ".gz"))
        {
            std::cout << "Processing: " << (it->path().string()) << std::endl;
            read_file(it->path());
        }
    }

}

int main(int argc, char *argv[])
{
    read_directory();
    return 0;
}
