#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

// Constants
const bool useFP16 = false;


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

struct InferenceResult
{
    int32_t true_positives;
    int32_t true_negatives;
    int32_t false_positives;
    int32_t false_negatives;

    void reset()
    {
        true_positives = 0;
        true_negatives = 0;
        false_positives = 0;
        false_negatives = 0;
    }

    int32_t n_events()
    {
        return true_positives + true_negatives + false_positives + false_negatives;
    }

    float accuracy()
    {
        return float(true_positives + true_negatives)/(float)n_events();
    }
};

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
//        if (severity <= nvinfer1::ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
} ghost_logger;

class NetworkInputDescriptor
{
    public:

        virtual ~NetworkInputDescriptor();

        // Branch names + _i or _f for integer/float branches
        virtual std::vector<std::string> get_branch_names() const=0;

        // Size of the host input buffers needed to read the ROOT file
        virtual unsigned host_buffer_size() const=0;

        // Size of the host(+device) input buffers needed for the NN
        virtual unsigned device_buffer_size() const=0;

        // Do the magic mapping the root file variables to the NN input
        virtual void fill_host_buffer(const void* root_buffer, void* host_buffer) const=0;
};

NetworkInputDescriptor::~NetworkInputDescriptor(){}

class GhostNetworkInputDescriptor: public NetworkInputDescriptor
{
    public:

        GhostNetworkInputDescriptor();

        ~GhostNetworkInputDescriptor();

        std::vector<std::string> get_branch_names() const;

        unsigned host_buffer_size() const;

        unsigned device_buffer_size() const;

        // Do the magic mapping the root file variables to the NN input
        void fill_host_buffer(const void* root_buffer, void* host_buffer) const;
};

GhostNetworkInputDescriptor::GhostNetworkInputDescriptor(){}

GhostNetworkInputDescriptor::~GhostNetworkInputDescriptor(){}

std::vector<std::string> GhostNetworkInputDescriptor::get_branch_names() const
{
    return std::vector<std::string>({"x_f", "y_f", "tx_f", "ty_f", "best_qop_f", "best_pt_f", "kalman_ip_chi2_f", 
                                        "kalman_docaz_f", "chi2_f", "chi2V_f", "chi2UT_f", "chi2T_f", 
                                        "ndof_i", "ndofV_i", "ndofT_i", "nUT_i"});
}

unsigned GhostNetworkInputDescriptor::host_buffer_size() const
{
    return 12 * sizeof(float) + 4 * sizeof(int32_t);
}

unsigned GhostNetworkInputDescriptor::device_buffer_size() const
{
    return 17 * sizeof(float);
}

void GhostNetworkInputDescriptor::fill_host_buffer(const void* root_buffer, void* host_buffer) const
{
    ((float*)host_buffer)[0] = 1./ std::abs(((const float*) root_buffer)[4]);
    std::memcpy(host_buffer + sizeof(float), root_buffer, 12 * sizeof(float));
    const int32_t* int_buffer = (const int32_t*)(root_buffer + 12 * sizeof(float));
    ((float*)host_buffer)[13] = (float)(int_buffer[0]);
    ((float*)host_buffer)[14] = (float)(int_buffer[1]);
    ((float*)host_buffer)[15] = (float)(int_buffer[2]);
    ((float*)host_buffer)[16] = (float)(int_buffer[3]);
}

class InputDataProvider
{
    public:

        InputDataProvider(const NetworkInputDescriptor* network);

        virtual ~InputDataProvider();

        const NetworkInputDescriptor* network_input() const;

        virtual bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)=0;

        virtual int32_t* read_events(const int32_t event, const int32_t batchSize)=0;

        virtual bool fill_device_buffer(void* buffer, const int32_t batchSize)=0;

    protected:

        bool open(  const std::string& rootFile, const std::string treeName, 
                    const std::vector<std::string>& branchNames, void* inputBuffer, void* outputBuffer);

        bool close();

        const NetworkInputDescriptor* mNetworkDescriptor;
        TFile* mRootFile;
        TTree* mEventTree;
    
};

InputDataProvider::InputDataProvider(const NetworkInputDescriptor* network): mNetworkDescriptor(network), mRootFile(nullptr), mEventTree(nullptr){}

InputDataProvider::~InputDataProvider()
{
    close();
}

const NetworkInputDescriptor* InputDataProvider::network_input() const
{
    return mNetworkDescriptor;
}

bool InputDataProvider::open(const std::string& rootFile, const std::string treeName, const std::vector<std::string>& branchNames, 
                        void* inputBuffer, void* outputBuffer)
{
    close();
    mRootFile = new TFile(rootFile.c_str());
    mEventTree = (TTree*)mRootFile->Get(treeName.c_str());
    int32_t offset = 0;
    for(int32_t i = 0; i < branchNames.size(); ++i)
    {
        auto typechar = branchNames[i].back();
        auto branchName = branchNames[i].substr(0, branchNames[i].size()-2);
        if(typechar == 'i')
        {
            mEventTree->SetBranchAddress(branchName.c_str(), (inputBuffer + offset));
            offset += sizeof(int32_t);
        }
        else if(typechar == 'f')
        {
            mEventTree->SetBranchAddress(branchName.c_str(), (inputBuffer + offset));
            offset += sizeof(float);
        }
        else
        {
            throw 1;
        }
    }

    mEventTree->SetBranchAddress("ghost", (unsigned*)outputBuffer);
}

bool InputDataProvider::close()
{
    if(mEventTree != nullptr)
    {
        delete mEventTree;
    }
    if(mRootFile != nullptr)
    {
        mRootFile->Close();
        delete mRootFile;
    }
}

class FileInputDataProvider: public InputDataProvider
{
    public:

        FileInputDataProvider(const NetworkInputDescriptor* network);

        ~FileInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize);

        int32_t* read_events(const int32_t event, const int32_t batchSize);

        bool fill_device_buffer(void* buffer, const int32_t batchSize);

    private:

        void* mInputBuffer;
        void* mOutputBuffer;
        void* mInputTransferBuffer;
        void* mOutputTransferBuffer;
};

FileInputDataProvider::FileInputDataProvider(const NetworkInputDescriptor* network):InputDataProvider(network),
    mInputBuffer(nullptr), mOutputBuffer(nullptr), mInputTransferBuffer(nullptr), mOutputTransferBuffer(nullptr){}

FileInputDataProvider::~FileInputDataProvider()
{
    if(mInputBuffer != nullptr)
    {
        cudaFreeHost(mInputBuffer);
    }
    if(mOutputBuffer != nullptr)
    {
        cudaFreeHost(mOutputBuffer);
    }
    if(mInputTransferBuffer != nullptr)
    {
        cudaFreeHost(mInputTransferBuffer);
    }
    if(mOutputTransferBuffer != nullptr)
    {
        cudaFreeHost(mOutputTransferBuffer);
    }
}


bool FileInputDataProvider::load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)
{
    auto branchNames = this->mNetworkDescriptor->get_branch_names();

    if(mInputBuffer != nullptr)
    {
        cudaFreeHost(mInputBuffer);
    }
    cudaHostAlloc(&mInputBuffer, this->mNetworkDescriptor->host_buffer_size(), cudaHostAllocDefault);

    if(mInputTransferBuffer != nullptr)
    {
        cudaFreeHost(mInputTransferBuffer);
    }
    cudaHostAlloc(&mInputTransferBuffer, this->mNetworkDescriptor->device_buffer_size() * batchSize, cudaHostAllocDefault);

    if(mOutputBuffer != nullptr)
    {
        cudaFreeHost(mOutputBuffer);
    }
    cudaHostAlloc(&mOutputBuffer, sizeof(int32_t), cudaHostAllocDefault);

    if(mOutputTransferBuffer != nullptr)
    {
        cudaFreeHost(mOutputTransferBuffer);
    }
    cudaHostAlloc(&mOutputTransferBuffer, sizeof(int32_t) * batchSize, cudaHostAllocDefault);

    this->open(rootFile, treeName, branchNames, (void*)mInputBuffer, (void*)mOutputBuffer);

    return true;
}

int32_t* FileInputDataProvider::read_events(int32_t event, const int32_t batchSize)
{
    if(mEventTree == nullptr)
    {
        return nullptr;
    }
    int32_t stride = this->mNetworkDescriptor->device_buffer_size();
    for(int32_t i = 0; i < batchSize ; ++i)
    {
        mEventTree->GetEntry(event + i);
        this->mNetworkDescriptor->fill_host_buffer(mInputBuffer, mInputTransferBuffer + stride * i);
        ((int32_t*)mOutputTransferBuffer)[i] = ((unsigned*)mOutputBuffer)[0];
    }
    return (int32_t*)mOutputTransferBuffer;
}

bool FileInputDataProvider::fill_device_buffer(void* inputBufferDevice, const int32_t batchSize)
{
    cudaMemcpy(inputBufferDevice, mInputTransferBuffer, 
        this->mNetworkDescriptor->device_buffer_size() * batchSize, cudaMemcpyHostToDevice);
    return true;
}

class GPUInputDataProvider: public InputDataProvider
{
    public:

        GPUInputDataProvider(const NetworkInputDescriptor* network);

        ~GPUInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize);

        int32_t* read_events(const int32_t event, const int32_t batchSize);

        bool fill_device_buffer(void* buffer, const int32_t batchSize);

    private:

        void* mInputBuffer;
        void* mOutputBuffer;
        std::vector<int32_t> truths;
        int32_t index;
        int32_t stride;
};

GPUInputDataProvider::GPUInputDataProvider(const NetworkInputDescriptor* network): InputDataProvider(network), 
mInputBuffer(nullptr), mOutputBuffer(nullptr), index(-1){}

GPUInputDataProvider::~GPUInputDataProvider()
{
    if(mInputBuffer != nullptr)
    {
        cudaFree(mInputBuffer);
        mInputBuffer = nullptr;
    }
    if(mOutputBuffer != nullptr)
    {
        cudaFree(mOutputBuffer);
        mOutputBuffer = nullptr;
    }
    truths.clear();
}

bool GPUInputDataProvider::load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)
{
    auto nevts = 10000;
    stride = this->mNetworkDescriptor->device_buffer_size();
    FileInputDataProvider fileInput(this->mNetworkDescriptor);
    cudaMalloc(&mInputBuffer, nevts * stride);
    fileInput.load(rootFile, treeName, 1);
    for(int32_t i = 0; i < nevts; ++i)
    {
        auto result = fileInput.read_events(i, 1);
        truths.push_back(result[0]);
        fileInput.fill_device_buffer(mInputBuffer + i * stride, 1);
    }
    return true;
}

int32_t* GPUInputDataProvider::read_events(int32_t event, const int32_t batchSize)
{
    index = event;
    return truths.data() + index;
}

bool GPUInputDataProvider::fill_device_buffer(void* buffer, const int32_t batchSize)
{
    cudaMemcpy(buffer, mInputBuffer + index * stride * batchSize, stride * batchSize, cudaMemcpyDeviceToDevice);
}



class GhostDetection
{
    public:
    
        GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider);

        bool build(const int32_t max_batch_size);

        bool initialize(const std::string& rootFile, const std::string& treeName, int32_t batchSize);

        bool infer(const int32_t nevent, const int32_t batchSize, InferenceResult& result);

    private:

        std::string mEngineFilename;

        InputDataProvider* mDataProvider;

        std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> mEngine;

        std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> mContext;
                
        const int32_t mInputSize, mOutputSize;

        int32_t mBatchSize;
        void* mInputBufferDevice;
        void* mOutputBufferDevice;
};

GhostDetection::GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider): mEngineFilename(engineFilename), 
mDataProvider(dataProvider), mEngine(nullptr), mContext(nullptr), mInputSize(17), mOutputSize(1), mBatchSize(1), mInputBufferDevice(nullptr), 
mOutputBufferDevice(nullptr){}

bool GhostDetection::initialize(const std::string& rootFile, const std::string& treeName, int32_t batchSize)
{
    mBatchSize = batchSize;
    mDataProvider->load(rootFile, treeName, batchSize);
    if(mInputBufferDevice != nullptr)
    {
        cudaFree(mInputBufferDevice);
        mInputBufferDevice = nullptr;
    }
    cudaMalloc(&mInputBufferDevice, mDataProvider->network_input()->host_buffer_size() * batchSize);

    if(mOutputBufferDevice != nullptr)
    {
        cudaFree(mOutputBufferDevice);
        mOutputBufferDevice = nullptr;
    }
    cudaMalloc(&mOutputBufferDevice, sizeof(float) * batchSize);
    mContext->setOptimizationProfile(0);
    mContext->setBindingDimensions(0, nvinfer1::Dims2(batchSize, mInputSize));
    return true;
}

bool GhostDetection::build(const int32_t maxBatchSize)
{
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder, InferDeleter>(nvinfer1::createInferBuilder(ghost_logger));
    if (!builder)
    {
        return false;
    }
    builder->setMaxBatchSize(maxBatchSize);

    // If necessary check that the GPU supports lower precision
    if ( useFP16 && !builder->platformHasFastFp16() )
    {
        return false;
    }

    // Create (empty) network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    // Create config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    config->setMaxWorkspaceSize(1024 * (1 << 20));
    if ( useFP16 )
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, mInputSize));
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(maxBatchSize/4, mInputSize));
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(maxBatchSize, mInputSize));

    config->addOptimizationProfile(profile);

    // Add profile stream to config
/*    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> profileStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        profileStream.reset(nullptr);
    }
    config->setProfileStream(*profileStream);*/

    // Define the ONNX parser with the network it should write to
    auto parser = std::unique_ptr<nvonnxparser::IParser, InferDeleter>(nvonnxparser::createParser(*network, ghost_logger));
    if (!parser)
    {
        return false;
    }

    // Parse the ONNX input network file
    auto parsed = parser->parseFromFile(mEngineFilename.c_str(), 0);
    if (!parsed)
    {
        return false;
    }

    // Do the actual building of the network
    std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    // Create inference runtime
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{nvinfer1::createInferRuntime(ghost_logger)};
    if (!runtime)
    {
        return false;
    }

    // Create engine
    mEngine = std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    mContext = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(mEngine->createExecutionContext());
    if (!mContext)
    {
        return false;
    }

    return true;
}


bool GhostDetection::infer(int32_t nevent, int32_t batchSize, InferenceResult& result)
{
    auto truths = mDataProvider->read_events(nevent, batchSize);
    mDataProvider->fill_device_buffer(mInputBufferDevice, batchSize);
    void* buffers[2];
    buffers[0] = mInputBufferDevice;
    buffers[1] = mOutputBufferDevice;

    // Do inference, batch size 1
    bool status = mContext->executeV2((void* const*)buffers);
    if (!status)
    {
        return false;
    }

    std::vector<float> prediction(batchSize);
 
    // Memcpy from host input buffers to device input buffers
    cudaMemcpy((void*)prediction.data(), mOutputBufferDevice, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);

    for(int32_t i = 0; i < batchSize; ++i)
    {
        int32_t pred = prediction[i] < 0.5 ? 0 : 1;
        if(truths[i] == pred)
        {
            result.true_positives += truths[i];
            result.true_negatives += (1 - truths[i]);
        }
        else
        {
            result.false_positives += pred;
            result.false_negatives += truths[i];
        }
    }

    return true;
}


int main(int argc, char* argv[])
{
    NetworkInputDescriptor* networkDescriptor = new GhostNetworkInputDescriptor();
    int offset = 1;
    InputDataProvider* inputDataProvider = nullptr;
    std::vector<int> batches = {1};
    if(argc > 0)
    {
        if(std::string(argv[1]) == "-f")
        {
            inputDataProvider = new FileInputDataProvider(networkDescriptor);
            offset = 2;
        }
        if (std::string(argv[1]) == "-g")
        {
            inputDataProvider = new GPUInputDataProvider(networkDescriptor);
            offset = 2;
        }
        for(auto i = offset; i < argc; ++i)
        {
            batches.push_back(std::stoi(argv[i]));
        }
    }
    if(inputDataProvider == nullptr)
    {
        inputDataProvider = new GPUInputDataProvider(networkDescriptor);
    }
    GhostDetection ghostinfer("../data/ghost_nn.onnx", inputDataProvider);
    bool retVal = true;
    std::sort(batches.begin(), batches.end());
    retVal = ghostinfer.build(batches.back());
    if ( !retVal )
    {
        std::cerr << "Build was not successful." << std::endl;
        return -1;
    }

    InferenceResult result;
    for(auto it = batches.begin(); it != batches.end(); ++it)
    {
        std::cout<<"Benchmarking batch size "<<*it<<"..."<<std::endl;

        result.reset();

        retVal = ghostinfer.initialize("../data/PrCheckerPlots.root","kalman_validator/kalman_ip_tree", *it);
        if ( !retVal )
        {
            std::cerr << "Initialization was not successful." << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(int32_t i = 0; i < 10000; i+=(*it))
        {
            ghostinfer.infer(i, *it, result);
        }
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//        if(it == batches.begin())
//        {
            std::cout<<".............................................."<<std::endl;
            std::cout<<"No. true positives:  "<<result.true_positives<<std::endl;
            std::cout<<"No. true negatives:  "<<result.true_negatives<<std::endl;
            std::cout<<"No. false positives: "<<result.false_positives<<std::endl;
            std::cout<<"No. false negatives: "<<result.false_negatives<<std::endl;
            std::cout<<".............................................."<<std::endl;
            std::cout<<"accuracy:" <<result.accuracy()<<std::endl<<std::endl<<std::endl;
//        }
        std::cout<<"Duration: "<<duration.count()<<" microsec."<<std::endl;

    }

    delete inputDataProvider;
    delete networkDescriptor;
}
