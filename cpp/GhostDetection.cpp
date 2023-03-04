#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>
#include <utility>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

// Constants
const bool useFP16 = false;
const bool useINT8 = false;

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

class VariableSet
{
    public:

        virtual ~VariableSet();

        // Branch names + _i or _f for integer/float branches
        virtual std::vector<std::string> get_branch_names() const=0;

        // Size of the host input buffers needed to read the ROOT file
        virtual unsigned host_buffer_size() const=0;

        // Size of the host(+device) input buffers needed for the NN
        virtual unsigned device_buffer_size() const=0;

        // Do the magic mapping the root file variables to the NN input
        virtual void fill_host_buffer(const void* root_buffer, void* host_buffer) const=0;

        // No events in the file
        virtual unsigned n_events() const=0;
};

VariableSet::~VariableSet(){}

class GhostVariableSet: public VariableSet
{
    public:

        GhostVariableSet();

        ~GhostVariableSet();

        std::vector<std::string> get_branch_names() const;

        unsigned host_buffer_size() const;

        unsigned device_buffer_size() const;

        void fill_host_buffer(const void* root_buffer, void* host_buffer) const;

        virtual unsigned n_events() const;
};

GhostVariableSet::GhostVariableSet(){}

GhostVariableSet::~GhostVariableSet(){}

std::vector<std::string> GhostVariableSet::get_branch_names() const
{
    return std::vector<std::string>({"x_f", "y_f", "tx_f", "ty_f", "best_qop_f", "best_pt_f", "kalman_ip_chi2_f", 
                                        "kalman_docaz_f", "chi2_f", "chi2V_f", "chi2UT_f", "chi2T_f", 
                                        "ndof_i", "ndofV_i", "ndofT_i", "nUT_i"});
}

unsigned GhostVariableSet::host_buffer_size() const
{
    return 12 * sizeof(float) + 4 * sizeof(int32_t);
}

unsigned GhostVariableSet::device_buffer_size() const
{
    return 17 * sizeof(float);
}

void GhostVariableSet::fill_host_buffer(const void* root_buffer, void* host_buffer) const
{
    ((float*)host_buffer)[0] = 1./ std::abs(((const float*) root_buffer)[4]);
    std::memcpy(host_buffer + sizeof(float), root_buffer, 12 * sizeof(float));
    const int32_t* int_buffer = (const int32_t*)(root_buffer + 12 * sizeof(float));
    ((float*)host_buffer)[13] = (float)(int_buffer[0]);
    ((float*)host_buffer)[14] = (float)(int_buffer[1]);
    ((float*)host_buffer)[15] = (float)(int_buffer[2]);
    ((float*)host_buffer)[16] = (float)(int_buffer[3]);
}

unsigned GhostVariableSet::n_events() const
{
    return 10000;
}

class InputDataProvider
{
    public:

        InputDataProvider(const VariableSet* variableSet);

        virtual ~InputDataProvider();

        const VariableSet* input_variables() const;

        virtual bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)=0;

        virtual bool read_events(const int32_t event, const int32_t batchSize, void* truth_buffer)=0;

        virtual bool fill_device_buffer(void* buffer, const int32_t batchSize)=0;

    protected:

        bool open(  const std::string& rootFile, const std::string treeName, 
                    const std::vector<std::string>& branchNames, void* inputBuffer, void* outputBuffer);

        bool close();

        const VariableSet* mVariableSet;
        TFile* mRootFile;
        TTree* mEventTree;
    
};

InputDataProvider::InputDataProvider(const VariableSet* network): mVariableSet(network), mRootFile(nullptr), mEventTree(nullptr){}

InputDataProvider::~InputDataProvider()
{
    close();
}

const VariableSet* InputDataProvider::input_variables() const
{
    return mVariableSet;
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

        FileInputDataProvider(const VariableSet* network);

        ~FileInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize);

        bool read_events(const int32_t event, const int32_t batchSize, void* truth_buffer);

        bool fill_device_buffer(void* buffer, const int32_t batchSize);

    private:

        void* mInputBuffer;
        void* mOutputBuffer;
        void* mInputTransferBuffer;
};

FileInputDataProvider::FileInputDataProvider(const VariableSet* variableSet):InputDataProvider(variableSet),
    mInputBuffer(nullptr), mOutputBuffer(nullptr), mInputTransferBuffer(nullptr){}

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
}


bool FileInputDataProvider::load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)
{
    auto branchNames = mVariableSet->get_branch_names();

    if(mInputBuffer != nullptr)
    {
        cudaFreeHost(mInputBuffer);
        mInputBuffer = nullptr;
    }
    cudaHostAlloc(&mInputBuffer, mVariableSet->host_buffer_size(), cudaHostAllocDefault);

    if(mInputTransferBuffer != nullptr)
    {
        cudaFreeHost(mInputTransferBuffer);
        mInputTransferBuffer = nullptr;
    }
    cudaHostAlloc(&mInputTransferBuffer, mVariableSet->device_buffer_size() * batchSize, cudaHostAllocDefault);

    if(mOutputBuffer != nullptr)
    {
        cudaFreeHost(mOutputBuffer);
        mOutputBuffer = nullptr;
    }
    cudaHostAlloc(&mOutputBuffer, sizeof(int32_t), cudaHostAllocDefault);

    this->open(rootFile, treeName, branchNames, (void*)mInputBuffer, (void*)mOutputBuffer);

    return true;
}

bool FileInputDataProvider::read_events(int32_t event, const int32_t batchSize, void* truth_buffer)
{
    if(mEventTree == nullptr)
    {
        return false;
    }
    int32_t stride = mVariableSet->device_buffer_size();
    for(int32_t i = 0; i < batchSize ; ++i)
    {
        mEventTree->GetEntry(event + i);
        mVariableSet->fill_host_buffer(mInputBuffer, mInputTransferBuffer + stride * i);
        ((int32_t*)truth_buffer)[i] = ((unsigned*)mOutputBuffer)[0];
    }
    return true;
}

bool FileInputDataProvider::fill_device_buffer(void* inputBufferDevice, const int32_t batchSize)
{
    return cudaMemcpy(inputBufferDevice, mInputTransferBuffer, mVariableSet->device_buffer_size() * batchSize, cudaMemcpyHostToDevice);
}

class GPUInputDataProvider: public InputDataProvider
{
    public:

        GPUInputDataProvider(const VariableSet* variableSet);

        ~GPUInputDataProvider();

        bool load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize);

        bool read_events(const int32_t event, const int32_t batchSize, void* truth_buffer);

        bool fill_device_buffer(void* buffer, const int32_t batchSize);

    private:

        void* mInputBuffer;
        void* mTruths;
        int32_t index;
};

GPUInputDataProvider::GPUInputDataProvider(const VariableSet* variableSet): InputDataProvider(variableSet), 
mInputBuffer(nullptr), mTruths(nullptr), index(-1){}

GPUInputDataProvider::~GPUInputDataProvider()
{
    if(mInputBuffer != nullptr)
    {
        cudaFree(mInputBuffer);
        mInputBuffer = nullptr;
    }
    if(mTruths != nullptr)
    {
        cudaFreeHost(mTruths);
        mTruths = nullptr;
    }
}

bool GPUInputDataProvider::load(const std::string& rootFile, const std::string& treeName, const int32_t batchSize)
{
    auto sampleSize = this->mVariableSet->n_events();
    auto eventSize = this->mVariableSet->device_buffer_size();

    if(mInputBuffer != nullptr)
    {
        cudaFree(mInputBuffer);
        mInputBuffer = nullptr;
    }
    cudaMalloc(&mInputBuffer, sampleSize * eventSize);

    if(mTruths != nullptr)
    {
        cudaFreeHost(mTruths);
        mTruths = nullptr;
    }
    cudaHostAlloc(&mTruths, sampleSize * sizeof(int32_t),  cudaHostAllocDefault);

    FileInputDataProvider fileInput(this->mVariableSet);
    fileInput.load(rootFile, treeName, 1);
    for(int32_t i = 0; i < sampleSize; ++i)
    {
        auto result = fileInput.read_events(i, 1, mTruths + i * sizeof(int32_t));
        fileInput.fill_device_buffer(mInputBuffer + i * eventSize, 1);
    }
    return true;
}

bool GPUInputDataProvider::read_events(int32_t event, const int32_t batchSize, void* truth_buffer)
{
    index = event;
    return cudaMemcpy(truth_buffer, mTruths + index * sizeof(int32_t), batchSize * sizeof(int32_t), cudaMemcpyHostToHost);
}

bool GPUInputDataProvider::fill_device_buffer(void* buffer, const int32_t batchSize)
{
    auto eventSize = this->mVariableSet->device_buffer_size();
    cudaMemcpy(buffer, mInputBuffer + index * eventSize, eventSize * batchSize, cudaMemcpyDeviceToDevice);
    return true;
}



class GhostDetection
{
    public:
    
        GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider);

        bool build(const int32_t max_batch_size);

        bool initialize(const std::string& rootFile, const std::string& treeName, int32_t batchSize);

        bool infer_batch(const int32_t nevent, const int32_t batchSize);

        InferenceResult infer_sample(const int32_t batchSize);

        InferenceResult collect_results() const;

    private:

        bool setDynamicRange(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>&);

        void setLayerPrecision(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>&);

        std::string mEngineFilename;

        InputDataProvider* mDataProvider;

        std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> mEngine;

        std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> mContext;

        void* mInputBufferDevice;
        void* mOutputBufferDevice;
        void* mTruthBufferHost;
};

GhostDetection::GhostDetection(const std::string& engineFilename, InputDataProvider* dataProvider): mEngineFilename(engineFilename), 
mDataProvider(dataProvider), mEngine(nullptr), mContext(nullptr), mInputBufferDevice(nullptr), mOutputBufferDevice(nullptr),
mTruthBufferHost(nullptr){}

bool GhostDetection::initialize(const std::string& rootFile, const std::string& treeName, int32_t batchSize)
{
    mDataProvider->load(rootFile, treeName, batchSize);
    auto sampleSize = this->mDataProvider->input_variables()->n_events();

    if(mInputBufferDevice != nullptr)
    {
        cudaFree(mInputBufferDevice);
        mInputBufferDevice = nullptr;
    }
    cudaMalloc(&mInputBufferDevice, mDataProvider->input_variables()->device_buffer_size() * batchSize);

    if(mOutputBufferDevice != nullptr)
    {
        cudaFree(mOutputBufferDevice);
        mOutputBufferDevice = nullptr;
    }
    cudaMalloc(&mOutputBufferDevice, sizeof(float) * sampleSize);

    if(mTruthBufferHost != nullptr)
    {
        cudaFreeHost(mTruthBufferHost);
        mTruthBufferHost = nullptr;
    }
    cudaHostAlloc(&mTruthBufferHost, sizeof(int32_t) * sampleSize, cudaHostAllocDefault);

    mContext->setOptimizationProfile(0);
    auto inputSize = mDataProvider->input_variables()->device_buffer_size() / sizeof(float);
    mContext->setBindingDimensions(0, nvinfer1::Dims2(batchSize, inputSize));
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

    if( !useINT8 )
    {
    // If necessary check that the GPU supports lower precision
    if ( useFP16 && !builder->platformHasFastFp16() )
    {
        return false;
    }
    if ( useINT8 )
    {
        if ( builder->platformHasFastInt8() )
        {
            builder->setInt8Mode(true);
        }
        else
        {
            return false;
        }
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
    if ( !useINT8 )
    {
        if ( useFP16 )
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }
    else
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
        config->setInt8Calibrator(nullptr);

        // force layer to execute with required precision
        setLayerPrecision(network);

        // set INT8 Per Tensor Dynamic range
        if (!setDynamicRange(network))
        {
            std::cerr << "Unable to set per-tensor dynamic range." << std::endl;
            return false;
        }
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    auto inputSize = mDataProvider->input_variables()->device_buffer_size() / sizeof(float);
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, inputSize));
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(maxBatchSize/4, inputSize));
    profile->setDimensions("dense_input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(maxBatchSize, inputSize));

    config->addOptimizationProfile(profile);

    // Add profile stream to config
/*    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> profileStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        profileStream.reset(nullptr);
    }
    config->setProfileStream(*profileStream);
*/

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


bool GhostDetection::setDynamicRange(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>& network)
{
    std::cerr << "Setting Per Tensor Dynamic Range" << std::endl;

    // set dynamic range for network input tensors
    network->getInput(0)->setDynamicRange(-10., 10.);

    network->getInput(1)->setDynamicRange(-10., 10.);
    network->getInput(2)->setDynamicRange(-0.3, 0.3);
    network->getInput(3)->setDynamicRange(-0.3, 0.3);
    network->getInput(4)->setDynamicRange(0., 1.);
    network->getInput(5)->setDynamicRange(0., 15000.);
    network->getInput(6)->setDynamicRange(-0.5, 10000.5);
    network->getInput(7)->setDynamicRange(-0.5, 25.5);
    network->getInput(8)->setDynamicRange(0., 400.);
    network->getInput(9)->setDynamicRange(0., 150.);
    network->getInput(10)->setDynamicRange(0., 150.);
    network->getInput(11)->setDynamicRange(0., 150.);
    network->getInput(12)->setDynamicRange(0.,10.);
    network->getInput(13)->setDynamicRange(0.,10.);
    network->getInput(14)->setDynamicRange(0.,10.);
    network->getInput(15)->setDynamicRange(0.,10.);
    network->getInput(16)->setDynamicRange(0.,1.);


    // set dynamic range for layer output tensors
    network->getOutput(0)->setDynamicRange(0.,1.);

    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j)
        {
            if (lyr->getType() == nvinfer1::LayerType::kCONSTANT)
            {
                IConstantLayer* cLyr = static_cast<IConstantLayer*>(lyr);
                auto wts = cLyr->getWeights();
                double max = std::numeric_limits<double>::min();
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb)
                {
                    double val{};
                    switch (wts.type)
                    {
                        case nvinfer1::DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                        case nvinfer1::DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                        case nvinfer1::DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                        case nvinfer1::DataType::kHALF: val = static_cast<const half_float::half*>(wts.values)[wb]; break;
                        case nvinfer1::DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    }
                    max = std::max(max, std::abs(val));
                }
                if (!lyr->getOutput(j)->setDynamicRange(-max, max))
                {
                    return false;
                }
            }
            else
            {
                double max=128.; // completely random, we use this for perf testing only
                if (!lyr->getOutput(j)->setDynamicRange(-max, max))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

void GhostDetection::setLayerPrecision(std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter>& network)
{
    std::cerr << "Setting Per Layer Computation Precision" << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);

        std::string layerName = layer->getName();
        std::cout << "Layer: " << layerName << ". Precision: INT8" << std::endl;

        // Don't set the precision on non-computation layers as they don't support
        // int8.
        if (layer->getType() != nvinfer1::LayerType::kCONSTANT && layer->getType() != nvinfer1::LayerType::kCONCATENATION
            && layer->getType() != nvinfer1::LayerType::kSHAPE)
        {
            // set computation precision of the layer
            layer->setPrecision(nvinfer1::DataType::kINT8);
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            std::cout << "Tensor: " << tensorName << ". OutputType: INT8" << std::endl;
            // set output type of execution tensors and not shape tensors.
            if (layer->getOutput(j)->isExecutionTensor())
            {
                layer->setOutputType(j, nvinfer1::DataType::kINT8);
            }
        }
    }
}



InferenceResult GhostDetection::collect_results() const
{
    auto sampleSize = this->mDataProvider->input_variables()->n_events();
    std::vector<float> prediction(sampleSize);
 
    // Memcpy from host input buffers to device input buffers
    cudaMemcpy((void*)prediction.data(), mOutputBufferDevice, sizeof(float) * sampleSize, cudaMemcpyDeviceToHost);

    auto truths = (int32_t*)mTruthBufferHost;

    InferenceResult result;
    result.reset();

    for(int32_t i = 0; i < sampleSize; ++i)
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

    return result;
}


bool GhostDetection::infer_batch(const int32_t nevent, const int32_t batchSize)
{
    auto clipBatch = std::min((int32_t)(this->mDataProvider->input_variables()->n_events()) - nevent - 1, batchSize);

    mDataProvider->read_events(nevent, clipBatch, mTruthBufferHost + nevent * sizeof(int32_t));
    mDataProvider->fill_device_buffer(mInputBufferDevice, clipBatch);

    void* buffers[2];
    buffers[0] = mInputBufferDevice;
    buffers[1] = mOutputBufferDevice + nevent * sizeof(float);

    // Do inference
    return mContext->executeV2((void* const*)buffers);
}

InferenceResult GhostDetection::infer_sample(const int32_t batchSize)
{
    for(int32_t i = 0; i < mDataProvider->input_variables()->n_events(); i += batchSize)
    {
        infer_batch(i, batchSize);
    }
    return collect_results();
}


int main(int argc, char* argv[])
{
    VariableSet* inputVariables = new GhostVariableSet();
    int offset = 1;
    InputDataProvider* inputDataProvider = nullptr;
    std::vector<int> batches = {1};
    if(argc > 0)
    {
        if(std::string(argv[1]) == "-h" or std::string(argv[1]) == "--help")
        {
            std::cout<<"Usage: ./main -<f|g> <N1> <N2> <N3> ..."<<std::endl;
            std::cout<<"Options:"<<std::endl;
            std::cout<<"\t-f: read from ROOT file per batch and transfer to GPU"<<std::endl;
            std::cout<<"\t-g: read full ROOT file and transfer all data to GPU first (default)"<<std::endl;
            std::cout<<"\tN1 (int): first batch size to benchmark"<<std::endl;
            std::cout<<"\tN2 (int): second batch size to benchmark"<<std::endl;
            std::cout<<"\tN3 (int): third batch size to benchmark"<<std::endl;
            std::cout<<"\t..."<<std::endl;
            std::exit(0);
        }
        if(std::string(argv[1]) == "-f")
        {
            inputDataProvider = new FileInputDataProvider(inputVariables);
            offset = 2;
        }
        if (std::string(argv[1]) == "-g")
        {
            inputDataProvider = new GPUInputDataProvider(inputVariables);
            offset = 2;
        }
        for(auto i = offset; i < argc; ++i)
        {
            batches.push_back(std::stoi(argv[i]));
        }
    }
    if(inputDataProvider == nullptr)
    {
        inputDataProvider = new GPUInputDataProvider(inputVariables);
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

    for(auto it = batches.begin(); it != batches.end(); ++it)
    {
        std::cout<<"Benchmarking batch size "<<*it<<"..."<<std::endl;

        retVal = ghostinfer.initialize("../data/PrCheckerPlots.root","kalman_validator/kalman_ip_tree", *it);
        if ( !retVal )
        {
            std::cerr << "Initialization was not successful." << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        for(int32_t i = 0; i < inputDataProvider->input_variables()->n_events(); i+=(*it))
        {
            ghostinfer.infer_batch(i, *it);
        }
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        auto result = ghostinfer.collect_results();
        std::cout<<".............................................."<<std::endl;
        std::cout<<"No. true positives:  "<<result.true_positives<<std::endl;
        std::cout<<"No. true negatives:  "<<result.true_negatives<<std::endl;
        std::cout<<"No. false positives: "<<result.false_positives<<std::endl;
        std::cout<<"No. false negatives: "<<result.false_negatives<<std::endl;
        std::cout<<".............................................."<<std::endl;
        std::cout<<"accuracy:" <<result.accuracy()<<std::endl;
        std::cout<<"Duration: "<<duration.count()<<" microseconds"<<std::endl<<std::endl<<std::endl;
    }

    delete inputDataProvider;
    delete inputVariables;
}
