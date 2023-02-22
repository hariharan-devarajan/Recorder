#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "reader.h"
#include <parquet/arrow/writer.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <stdint-gcc.h>
#include <mpi.h>
#include <map>
#include <experimental/filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>

#include <codecvt>
namespace fs = std::experimental::filesystem;


std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

class Directory{
public:
    std::string directory;
    std::string hostname;
    std::string username;
    std::string app;
    int proc_id;
    double timestamp;
    Directory():hostname(), username(), app(), proc_id(0), timestamp(0.0) {} /* default constructor */
    Directory(const Directory &other)
            : hostname(other.hostname),
              username(other.username),
              app(other.app),
              proc_id(other.proc_id),
              timestamp(other.timestamp) {} /* copy constructor*/
    Directory(Directory &&other)
            : hostname(other.hostname),
              username(other.username),
              app(other.app),
              proc_id(other.proc_id),
              timestamp(other.timestamp){} /* move constructor*/

    Directory(std::string path){
        this->directory = path;
        std::vector<std::string> split_strings = split(path, '-');
        if (split_strings.size() >=4) {
            hostname = split_strings[0];
            username = split_strings[1];    
            app = split_strings[2];    
            for (int i = 3; i<split_strings.size() - 2;++i) {
                   app += "-" + split_strings[i];
            }
            proc_id = atoi(split_strings[split_strings.size() - 2].c_str());    
            timestamp = atof(split_strings[split_strings.size() - 1].c_str());    
        }
    }

    Directory& operator=(const Directory& other) {

        this->hostname = other.hostname;
        this->username = other.username;
        this->app = other.app;
        this->proc_id = other.proc_id;
        this->timestamp = other.timestamp;
        return *this;
    }


    bool operator==(const Directory& other) const{
        return this->timestamp == other.timestamp;
    }

    bool operator!=(const Directory& other) const{
        return this->timestamp != other.timestamp;
    }

    bool operator<(const Directory& other) const{
        return this->timestamp < other.timestamp;
    }

    bool operator>(const Directory& other) const{
        return this->timestamp > other.timestamp;
    }

    bool operator<=(const Directory& other) const{
        return this->timestamp <= other.timestamp;
    }

    bool operator>=(const Directory& other) const{
        return this->timestamp >= other.timestamp;
    }
};


struct ParquetWriter {
    Directory directory;
    int rank;
    char* base_file;
    int64_t index;
    arrow::Int32Builder categoryBuilder, rankBuilder, threadBuilder, levelBuilder;
    arrow::Int64Builder procBuilder,indexBuilder, sizeBuilder;
    arrow::FloatBuilder tstartBuilder, tendBuilder, durationBuilder, bandwidthBuilder, tmidBuilder;
    arrow::StringBuilder func_idBuilder, appBuilder, hostnameBuilder, filenameBuilder;

    std::shared_ptr<arrow::Array> indexArray, categoryArray, procArray, rankArray, threadArray, levelArray, tstartArray, tendArray,
            func_idArray, appArray,hostnameArray, sizeArray, durationArray, bandwidthArray, filenameArray, tmidArray;

    std::shared_ptr<arrow::Schema> schema;
    const int64_t chunk_size = 1024;
    const int64_t NUM_ROWS = 1024*1024*16; // 1B
    int64_t row_group = 0;
    ParquetWriter(char* _path);
    void finish();
    float max_tend;
    std::unordered_set<std::string> unique_filenames;
    std::unordered_set<std::string> unique_processes;
    uint64_t sum_transfer_size;
    double sum_bandwidth;
    uint64_t record_count;
};
ParquetWriter::ParquetWriter(char* _path):unique_filenames(),
                                max_tend(0), unique_processes(),
                                sum_transfer_size(0),sum_bandwidth(0),
                                record_count(0)   {
    row_group = 0;
    base_file = _path;
    schema = arrow::schema(
            {arrow::field("index", arrow::int64()), arrow::field("proc", arrow::int64()),
             arrow::field("rank", arrow::int32()),
             arrow::field("thread_id", arrow::int32()), arrow::field("cat", arrow::int32()),
             arrow::field("tstart", arrow::float32()), arrow::field("tend", arrow::float32()),
             arrow::field("func_id", arrow::utf8()), arrow::field("level", arrow::int32()),
             arrow::field("hostname", arrow::utf8()), arrow::field("app", arrow::utf8()),
             arrow::field("filename", arrow::utf8()), arrow::field("size", arrow::int64()),
              arrow::field("bandwidth", arrow::float32()), arrow::field("duration", arrow::float32()),
              arrow::field("tmid", arrow::float32())});
    index = 0;

    indexBuilder = arrow::Int64Builder();
    indexArray.reset();
    procBuilder = arrow::Int64Builder();
    procArray.reset();
    sizeBuilder = arrow::Int64Builder();
    sizeArray.reset();

    rankBuilder = arrow::Int32Builder();
    rankArray.reset();
    threadBuilder = arrow::Int32Builder();
    threadArray.reset();
    categoryBuilder = arrow::Int32Builder();
    categoryArray.reset();
    levelBuilder = arrow::Int32Builder();
    levelArray.reset();

    hostnameBuilder = arrow::StringBuilder();
    hostnameArray.reset();
    filenameBuilder = arrow::StringBuilder();
    filenameArray.reset();

    tstartBuilder = arrow::FloatBuilder();
    tstartArray.reset();
    tendBuilder = arrow::FloatBuilder();
    tendArray.reset();
    durationBuilder = arrow::FloatBuilder();
    durationArray.reset();
    bandwidthBuilder = arrow::FloatBuilder();
    bandwidthArray.reset();
    tmidBuilder = arrow::FloatBuilder();
    tmidArray.reset();


    func_idBuilder = arrow::StringBuilder();
    func_idArray.reset();
    appBuilder = arrow::StringBuilder();
    appArray.reset();
}

void ParquetWriter::finish(void) {
    indexBuilder.Finish(&indexArray);
    procBuilder.Finish(&procArray);
    sizeBuilder.Finish(&sizeArray);
    rankBuilder.Finish(&rankArray);
    threadBuilder.Finish(&threadArray);
    tstartBuilder.Finish(&tstartArray);
    tendBuilder.Finish(&tendArray);
    durationBuilder.Finish(&durationArray);
    bandwidthBuilder.Finish(&bandwidthArray);
    tmidBuilder.Finish(&tmidArray);
    func_idBuilder.Finish(&func_idArray);
    levelBuilder.Finish(&levelArray);
    hostnameBuilder.Finish(&hostnameArray);
    filenameBuilder.Finish(&filenameArray);
    appBuilder.Finish(&appArray);
    categoryBuilder.Finish(&categoryArray);

    auto table = arrow::Table::Make(schema, {indexArray, procArray, rankArray,threadArray,categoryArray,
                                             tstartArray, tendArray, func_idArray, levelArray,
                                             hostnameArray, appArray,
                                             filenameArray, sizeArray, bandwidthArray,
                                             durationArray, tmidArray });
    char path[256];
    sprintf(path, "%s_%d.parquet" , base_file, row_group);
    PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));
}

RecorderReader reader;

char* get_filename(Record* record) {
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE){
        return "";
    }
    const char* func_name = recorder_get_func_name(&reader, record);
    const char* open_condition = strstr(func_name, "open");
    const char* mpi_condition = strstr(func_name, "MPI");
    const char* close_condition = strstr(func_name, "close");
    const char* fread_condition = strstr(func_name, "fread");
    const  char* fwrite_condition = strstr(func_name, "fwrite");
    const  char* read_condition = strstr(func_name, "read");
    const  char* write_condition = strstr(func_name, "write");
    if(open_condition && !mpi_condition) return record->args[0];
    if(open_condition && mpi_condition) return record->args[1];
    if(close_condition) return record->args[0];
    if(read_condition && !fread_condition) return record->args[0];
    if(fread_condition) return record->args[3];
    if(write_condition && !fwrite_condition) return record->args[0];
    if(fwrite_condition) return record->args[3];
    return "";
}

int64_t get_size(Record* record) {
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE){
        return 0;
    }
    const char* func_name = recorder_get_func_name(&reader, record);
    const char* open_condition = strstr(func_name, "open");
    const char* mpi_condition = strstr(func_name, "MPI");
    const char* fread_condition = strstr(func_name, "fread");
    const char* close_condition = strstr(func_name, "close");
    const char* fwrite_condition = strstr(func_name, "fwrite");
    const char* read_condition = strstr(func_name, "read");
    const char* write_condition = strstr(func_name, "write");
    const char* readdir_condition = strstr(func_name, "readdir");
    if(read_condition && !fread_condition && !readdir_condition) return atol(record->args[2]);
    if(fread_condition) return atol(record->args[2]);
    if(write_condition && !fwrite_condition) return atol(record->args[2]);
    if(fwrite_condition) return atol(record->args[2]);
    return 0;
}

int64_t get_count(Record* record) {
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE){
        return 0;
    }
    const char* func_name = recorder_get_func_name(&reader, record);
    const  char* fread_condition = strstr(func_name, "fread");
    const  char* fwrite_condition = strstr(func_name, "fwrite");
    if(fread_condition) return atol(record->args[1]);
    if(fwrite_condition) return atol(record->args[1]);
    return 1;
}
void trim_utf8(std::string& hairy) {
    std::vector<bool> results;
    std::string smooth;
    size_t len = hairy.size();
    results.reserve(len);
    smooth.reserve(len);
    const unsigned char *bytes = (const unsigned char *) hairy.c_str();

    auto read_utf8 = [](const unsigned char *bytes, size_t len, size_t *pos) -> unsigned {
        int code_unit1 = 0;
        int code_unit2, code_unit3, code_unit4;

        if (*pos >= len) goto ERROR1;
        code_unit1 = bytes[(*pos)++];

        if (code_unit1 < 0x80) return code_unit1;
        else if (code_unit1 < 0xC2) goto ERROR1; // continuation or overlong 2-byte sequence
        else if (code_unit1 < 0xE0) {
            if (*pos >= len) goto ERROR1;
            code_unit2 = bytes[(*pos)++]; //2-byte sequence
            if ((code_unit2 & 0xC0) != 0x80) goto ERROR2;
            return (code_unit1 << 6) + code_unit2 - 0x3080;
        }
        else if (code_unit1 < 0xF0) {
            if (*pos >= len) goto ERROR1;
            code_unit2 = bytes[(*pos)++]; // 3-byte sequence
            if ((code_unit2 & 0xC0) != 0x80) goto ERROR2;
            if (code_unit1 == 0xE0 && code_unit2 < 0xA0) goto ERROR2; // overlong
            if (*pos >= len) goto ERROR2;
            code_unit3 = bytes[(*pos)++];
            if ((code_unit3 & 0xC0) != 0x80) goto ERROR3;
            return (code_unit1 << 12) + (code_unit2 << 6) + code_unit3 - 0xE2080;
        }
        else if (code_unit1 < 0xF5) {
            if (*pos >= len) goto ERROR1;
            code_unit2 = bytes[(*pos)++]; // 4-byte sequence
            if ((code_unit2 & 0xC0) != 0x80) goto ERROR2;
            if (code_unit1 == 0xF0 && code_unit2 <  0x90) goto ERROR2; // overlong
            if (code_unit1 == 0xF4 && code_unit2 >= 0x90) goto ERROR2; // > U+10FFFF
            if (*pos >= len) goto ERROR2;
            code_unit3 = bytes[(*pos)++];
            if ((code_unit3 & 0xC0) != 0x80) goto ERROR3;
            if (*pos >= len) goto ERROR3;
            code_unit4 = bytes[(*pos)++];
            if ((code_unit4 & 0xC0) != 0x80) goto ERROR4;
            return (code_unit1 << 18) + (code_unit2 << 12) + (code_unit3 << 6) + code_unit4 - 0x3C82080;
        }
        else goto ERROR1; // > U+10FFFF

        ERROR4:
        (*pos)--;
        ERROR3:
        (*pos)--;
        ERROR2:
        (*pos)--;
        ERROR1:
        return code_unit1 + 0xDC00;
    };

    unsigned c;
    size_t pos = 0;
    size_t pos_before;
    size_t inc = 0;
    bool valid;

    for (;;) {
        pos_before = pos;
        c = read_utf8(bytes, len, &pos);
        inc = pos - pos_before;
        if (!inc) break; // End of string reached.

        valid = false;

        if ( (                 c <= 0x00007F)
             ||   (c >= 0x000080 && c <= 0x0007FF)
             ||   (c >= 0x000800 && c <= 0x000FFF)
             ||   (c >= 0x001000 && c <= 0x00CFFF)
             ||   (c >= 0x00D000 && c <= 0x00D7FF)
             ||   (c >= 0x00E000 && c <= 0x00FFFF)
             ||   (c >= 0x010000 && c <= 0x03FFFF)
             ||   (c >= 0x040000 && c <= 0x0FFFFF)
             ||   (c >= 0x100000 && c <= 0x10FFFF) ) valid = true;

        if (c >= 0xDC00 && c <= 0xDCFF) {
            valid = false;
        }

        do results.push_back(valid); while (--inc);
    }

    size_t sz = results.size();
    for (size_t i = 0; i < sz; ++i) {
        if (results[i]) smooth.append(1, hairy.at(i));
    }

    hairy.swap(smooth);
}
void handle_one_record(Record* record, void* arg) {
    ParquetWriter *writer = (ParquetWriter*) arg;
    double duration = record->tend - record->tstart;
    char* filename = get_filename(record);
    int64_t size = get_size(record);
    int64_t count = get_count(record);
    size = size * count;
    double bandwidth = size * 1.0 / duration;
    double tmid = (record->tend + record->tstart) / 2.0;

    if (writer->max_tend < record->tend) writer->max_tend = record->tend;
    std::string file = std::string(filename);
    trim_utf8(file);
    writer->unique_filenames.emplace(file);
    writer->unique_processes.emplace(std::string(writer->directory.hostname) + "-" +
                                        std::to_string(writer->directory.proc_id) + "-" +
                                        std::to_string(writer->rank) + "-" +
                                        std::to_string(record->tid));
    writer->sum_transfer_size += size;
    writer->sum_bandwidth += bandwidth;
    writer->record_count ++;

    writer->durationBuilder.Append(duration);
    writer->filenameBuilder.Append(file);
    writer->sizeBuilder.Append(size);
    writer->bandwidthBuilder.Append(bandwidth);
    writer->tmidBuilder.Append(tmid);


    writer->procBuilder.Append(writer->directory.proc_id);
    writer->rankBuilder.Append(writer->rank);
    writer->threadBuilder.Append(record->tid);
    writer->tstartBuilder.Append(record->tstart);
    writer->tendBuilder.Append(record->tend);
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE){
        writer->func_idBuilder.Append(record->args[1]);
        record->arg_count = 1;
    }else {
        writer->func_idBuilder.Append(recorder_get_func_name(&reader, record));
    }
    writer->categoryBuilder.Append(cat);
    writer->levelBuilder.Append(record->level);
    writer->hostnameBuilder.Append(writer->directory.hostname);
    writer->appBuilder.Append(writer->directory.app);
    writer->index ++;
    writer->indexBuilder.Append(writer->index);
    if(writer->index % writer->NUM_ROWS == 0) {
        writer->indexBuilder.Finish(&writer->indexArray);
        writer->procBuilder.Finish(&writer->procArray);
        writer->sizeBuilder.Finish(&writer->sizeArray);
        writer->rankBuilder.Finish(&writer->rankArray);
        writer->threadBuilder.Finish(&writer->threadArray);
        writer->categoryBuilder.Finish(&writer->categoryArray);
        writer->tstartBuilder.Finish(&writer->tstartArray);
        writer->tendBuilder.Finish(&writer->tendArray);
        writer->durationBuilder.Finish(&writer->durationArray);
        writer->bandwidthBuilder.Finish(&writer->bandwidthArray);
        writer->tmidBuilder.Finish(&writer->tmidArray);
        writer->func_idBuilder.Finish(&writer->func_idArray);
        writer->levelBuilder.Finish(&writer->levelArray);
        writer->hostnameBuilder.Finish(&writer->hostnameArray);
        writer->filenameBuilder.Finish(&writer->filenameArray);
        writer->appBuilder.Finish(&writer->appArray);

        auto table = arrow::Table::Make(writer->schema, {writer->indexArray,writer->procArray, writer->rankArray,writer->threadArray, writer->categoryArray,
                                                         writer->tstartArray, writer->tendArray,
                                                         writer->func_idArray, writer->levelArray, writer->hostnameArray, writer->appArray,
                                                         writer->filenameArray, writer->sizeArray, writer->bandwidthArray,
                                                         writer->durationArray, writer->tmidArray});
        char path[256];
        sprintf(path, "%s_%d.parquet" , writer->base_file, writer->row_group);
        PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
        PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024*1024*128));

        writer->row_group++;
        writer->indexBuilder = arrow::Int64Builder();
        writer->indexArray.reset();
        writer->procBuilder = arrow::Int64Builder();
        writer->procArray.reset();
        writer->sizeBuilder = arrow::Int64Builder();
        writer->sizeArray.reset();
        writer->rankBuilder = arrow::Int32Builder();
        writer->rankArray.reset();
        writer->threadBuilder = arrow::Int32Builder();
        writer->threadArray.reset();
        writer->categoryBuilder = arrow::Int32Builder();
        writer->categoryArray.reset();
        writer->levelBuilder = arrow::Int32Builder();
        writer->levelArray.reset();
        writer->hostnameBuilder = arrow::StringBuilder();
        writer->hostnameArray.reset();
        writer->filenameBuilder = arrow::StringBuilder();
        writer->filenameArray.reset();

        writer->tstartBuilder = arrow::FloatBuilder();
        writer->tstartArray.reset();
        writer->tendBuilder = arrow::FloatBuilder();
        writer->tendArray.reset();
        writer->durationBuilder = arrow::FloatBuilder();
        writer->durationArray.reset();
        writer->bandwidthBuilder = arrow::FloatBuilder();
        writer->bandwidthArray.reset();
        writer->tmidBuilder = arrow::FloatBuilder();
        writer->tmidArray.reset();

        writer->func_idBuilder = arrow::StringBuilder();
        writer->func_idArray.reset();
        writer->appBuilder = arrow::StringBuilder();
        writer->appArray.reset();
    }
}

int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }



void process_rank(char* parquet_file_dir, int rank, Directory dir,ParquetWriter* writer) {


    CST cst;
    CFG cfg;
    writer->directory = dir;
    writer->rank = rank;
    recorder_read_cst(&reader, rank, &cst);
    recorder_read_cfg(&reader, rank, &cfg);
    recorder_decode_records(&reader, &cst, &cfg, handle_one_record, writer);
    printf("\r[Recorder] rank %d finished, unique call signatures: %d\n", rank, cst.entries);
    recorder_free_cst(&cst);
    recorder_free_cfg(&cfg);


}

int main(int argc, char **argv) {

    char parquet_file_dir[256], parquet_file_path[256];
    sprintf(parquet_file_dir, "%s/_parquet", argv[1]);
    mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    auto ordered_map = std::map<Directory, std::string>();
    if(mpi_rank == 0)
        mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    std::string base_path(argv[1]);
    bool steps = false;
    for (const auto & entry : fs::directory_iterator(argv[1])){
        if (fs::is_directory(entry)) {
            std::string dir_string{entry.path().u8string()};
            const size_t last_slash_idx = dir_string.rfind('/');
            std::string directory_name;
            if (std::string::npos != last_slash_idx) {
                directory_name = dir_string.substr(last_slash_idx + 1, std::string::npos - 1);
            }

            auto directory = Directory(directory_name);
            if (directory_name != "_parquet") {
                ordered_map.insert({directory, dir_string});
            }
            steps = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int step = 0;
    if (ordered_map.size() == 0) {
        auto dummy_directory_name = "localhost-user-app1-1-1.0";
        auto directory = Directory(dummy_directory_name);
        ordered_map.insert({directory, argv[1]});
    }
    int num_steps = ordered_map.size();
    int n = max(num_steps/mpi_size, 1);
    int start_step = n * mpi_rank;
    int end_step   = min(num_steps, n*(mpi_rank+1));
    int completed = 0;
    int prev=0;
    char parquet_filename_path[256];
    sprintf(parquet_filename_path, "%s/%d" , parquet_file_dir, mpi_rank);
    ParquetWriter writer(parquet_filename_path);
    for (auto x : ordered_map){
        if (reader.metadata.total_ranks == 1) {
            if (step >= start_step && step < end_step) {
                //printf("[Recorder] Converting workflow step %d of %d of rank 0 in %s by rank %d\n", step+1, num_steps, x.first.directory.c_str(), mpi_rank);
                recorder_init_reader(x.second.c_str(), &reader);
                process_rank(parquet_file_dir, 0, x.first, &writer);
                recorder_free_reader(&reader);
                completed++;
            }
        } else {
            recorder_init_reader(x.second.c_str(), &reader);
            // Each rank will process n files (n ranks traces)
            int n = max(reader.metadata.total_ranks/mpi_size, 1);
            int start_rank = n * mpi_rank;
            int end_rank   = min(reader.metadata.total_ranks, n*(mpi_rank+1));
            for(int rank = start_rank; rank < end_rank; rank++) {
                //printf("[Recorder] Converting workflow step %d of %d of rank %d in %s by rank %d\n", step+1, num_steps, rank, x.first.directory.c_str(), mpi_rank);
                process_rank(parquet_file_dir, rank, x.first, &writer);
            }
            recorder_free_reader(&reader);
            completed++;
        }
        step++;

        if(prev != completed && completed % 1 == 0) {
            printf("[Recorder] Completed %d of %d by rank %d\n", completed, num_steps/mpi_size, mpi_rank);
        }
        if (prev != completed) {
            prev = completed;
        }
    }
//    if (writer->max_tend < record->tend) writer->max_tend = record->tend;
//    writer->unique_filenames.emplace(std::string(filename));
//    writer->unique_processes.emplace(RankIndex(writer->directory.hostname, writer->directory.proc_id));
//    writer->sum_transfer_size += size;
//    writer->sum_bandwidth += bandwidth;
//    writer->record_count ++;
    using json = nlohmann::json;
    char rank_json[256];
    sprintf(rank_json, "%s/_parquet/rank_%d.json", argv[1],mpi_rank);
    json j;

    j["filenames"] = writer.unique_filenames;
    j["processes"] = writer.unique_processes;

    std::ofstream out(rank_json);
    out << j;
    out.close();
    printf("[Recorder] Written Json file by rank %d\n", mpi_rank);
    double global_max_tend;
    MPI_Reduce(&writer.max_tend, &global_max_tend, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    unsigned long total_transfer_size;
    MPI_Reduce(&writer.sum_transfer_size, &total_transfer_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    double total_bandwidth;
    MPI_Reduce(&writer.sum_bandwidth, &total_bandwidth, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    unsigned long total_count;
    MPI_Reduce(&writer.record_count, &total_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(mpi_rank == 0) {
        json j = {{"max_tend", global_max_tend},
                  {"total_bandwidth", total_bandwidth},
                  {"total_count", total_count},
                  {"total_transfer_size", total_transfer_size}};
        char global_json[256];
        sprintf(global_json, "%s/_parquet/global.json", argv[1]);
        std::ofstream out(global_json);
        out << j;
        out.close();
        printf("[Recorder] Written Global Json file by rank %d\n", mpi_rank);
    }

    writer.finish();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
