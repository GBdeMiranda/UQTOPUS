/*---------------------------------------------------------------------------*/

#include "onnxPolicy.H"
#include "error.H"
#include "IStringStream.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::onnxPolicy::onnxPolicy(const fileName& path, const label seed)
:
    path_(path),
    specHash_(),
    obsDim_(0),
    actDim_(0),
    rng_(seed),
    env_(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "uqtopusPolicy")),
    session_()
{
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    session_.reset(new Ort::Session(*env_, path_.c_str(), options));

    if (session_->GetInputCount() != 2 || session_->GetOutputCount() != 1)
    {
        FatalErrorInFunction
            << path_ << " has " << session_->GetInputCount() << " input(s) and "
            << session_->GetOutputCount() << " output(s). Contract 2.0 is "
            << "(observation, noise) -> action" << abort(FatalError);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session_->GetModelMetadata();

    auto entry = [&](const char* key) -> string
    {
        Ort::AllocatedStringPtr value =
            metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (value == nullptr)
        {
            FatalErrorInFunction
                << path_ << " has no '" << key << "' entry, so it was not "
                << "written by uqtopus.rl.export_policy" << abort(FatalError);
        }
        return string(value.get());
    };

    specHash_ = word(entry("uqtopus.spec_hash"));
    obsDim_ = readLabel(IStringStream(entry("uqtopus.obs_dim"))());
    actDim_ = readLabel(IStringStream(entry("uqtopus.act_dim"))());

    if (word(entry("uqtopus.distribution")) != "gaussian")
    {
        FatalErrorInFunction
            << path_ << " was exported for a " << entry("uqtopus.distribution")
            << " action, and only gaussian takes a normal draw"
            << abort(FatalError);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField> Foam::onnxPolicy::act
(
    const scalarField& observation,
    const bool deterministic
)
{
    if (observation.size() != obsDim_)
    {
        FatalErrorInFunction
            << "the graph expects " << obsDim_ << " observations, got "
            << observation.size() << abort(FatalError);
    }

    std::vector<float> obs(obsDim_);
    forAll(observation, i)
    {
        obs[i] = float(observation[i]);
    }

    // one draw per action component, Box-Muller over two uniforms, so this
    // uses only scalar01()
    std::vector<float> noise(actDim_, 0.0f);
    if (!deterministic)
    {
        for (label i = 0; i < actDim_; i++)
        {
            const scalar u1 = max(rng_.scalar01(), 1e-30);
            const scalar u2 = rng_.scalar01();
            noise[i] =
                float(sqrt(-2*log(u1))*cos(constant::mathematical::twoPi*u2));
        }
    }

    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> obsShape{1, int64_t(obsDim_)};
    const std::array<int64_t, 2> noiseShape{1, int64_t(actDim_)};

    std::vector<Ort::Value> inputs;
    inputs.push_back
    (
        Ort::Value::CreateTensor<float>
        (
            memory, obs.data(), obs.size(), obsShape.data(), obsShape.size()
        )
    );
    inputs.push_back
    (
        Ort::Value::CreateTensor<float>
        (
            memory, noise.data(), noise.size(),
            noiseShape.data(), noiseShape.size()
        )
    );

    const char* inputNames[] = {"observation", "noise"};
    const char* outputNames[] = {"action"};

    std::vector<Ort::Value> outputs = session_->Run
    (
        Ort::RunOptions{nullptr},
        inputNames, inputs.data(), inputs.size(),
        outputNames, 1
    );

    const float* data = outputs[0].GetTensorData<float>();
    tmp<scalarField> taction(new scalarField(actDim_));
    scalarField& action = taction.ref();
    forAll(action, i)
    {
        action[i] = scalar(data[i]);
    }
    return taction;
}


// ************************************************************************* //
