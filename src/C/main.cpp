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

#define SZ(v) (int)v.size()

int square(int x)
{
    return x*x;
}

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

template<class T>
int get_num_bounds(std::vector<T>& a)
{
    int num_bounds = 0;
    for(int i=0; i<SZ(a); i++)
    {
        if(a[i])
            num_bounds++;
    }
    return num_bounds;
}



template<class T>
void print_spacing_statistics(std::vector<T>& a)
{
    std::vector<int> positions;
    for(int i=0; i<SZ(a); i++)
    {
        if(a[i])
            positions.push_back(i);
    }
    // calc mean
    double tot_dist = 0;
    for(int i=0; i<SZ(positions)-1; i++){
        tot_dist += positions[i+1]-positions[i];
    }
    std::cout << "mean distance " << tot_dist / (SZ(positions)-1) << std::endl;
}


template <class T>
void print_distances(std::vector<std::vector<T>>& tracks, std::vector<std::string>& tissue_names)
{
    int num_tissues = SZ(tissue_names);
    /*
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
    }*/
    for(int i=0; i<num_tissues; i++)
    {
        std::cout << tissue_names[i] << std::endl;
        std::cout << get_num_bounds(tracks[i]) << std::endl;
        print_spacing_statistics(tracks[i]);
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

std::set<std::string> get_unique_celltypes(path entry)
{
    std::set<std::string> celltypes;
    // Decompress gzip file in memory and redirect to ss
    std::stringstream ss;
    ifstream file(entry, std::ios_base::in | std::ios_base::binary);
    filtering_streambuf<input> in;
    in.push(gzip_decompressor());
    in.push(file);
    boost::iostreams::copy(in, ss);
    std::string line;

    // read header file
    std::getline(ss, line);
    std::stringstream ssh(line);
    int num_tissues = -3;
    std::string token;
    while(ssh >> token)
    {
        if(num_tissues >= 0)
        {
            celltypes.insert(token);
        }
        num_tissues++;
    }
    return celltypes;
}

void read_directory()
{
    std::string dir_path = "../../data/chipseq_labels/";
    boost::filesystem::path p(dir_path);
    std::set<std::string> celltypes;

    for(directory_iterator it(dir_path); it!=directory_iterator(); ++it)
    {
        if (ends_with(it->path().string(), ".gz"))
        {
            std::cout << "Processing: " << (it->path().string()) << std::endl;
            read_file(it->path());

        }
    }

    std::cout << "number of distinct cell types " << celltypes.size() << std::endl;

}

int main(int argc, char *argv[])
{
    read_directory();
    return 0;
}
