#include <bound.h>


void print_bound_locations(path entry)
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
    std::vector<std::vector<int>> bound_locations(num_tissues, std::vector<double>());

}

void process_directory(std::string dir_path){
    boost::filesystem::path p(dir_path);
    std::set<std::string> celltypes;

    for(directory_iterator it(dir_path); it!=directory_iterator(); ++it)
    {
        if (ends_with(it->path().string(), ".gz"))
        {
            std::cout << "Processing: " << (it->path().string()) << std::endl;
            print_bound_locations(it->path());

        }
    }

    std::cout << "number of distinct cell types " << celltypes.size() << std::endl;
}
