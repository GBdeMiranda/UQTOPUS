/*---------------------------------------------------------------------------*/

#include "uqtopusController.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "interpolation.H"
#include "uniformDimensionedFields.H"
#include "OSspecific.H"

// * * * * * * * * * * * * * * * Static Data  * * * * * * * * * * * * * * * //

const Foam::word Foam::uqtopusController::registryName("uqtopusController");

const Foam::word Foam::uqtopusController::contractName("uqtopusPolicy");

namespace Foam
{
    defineTypeNameAndDebug(uqtopusController, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::uqtopusController::readContract(const dictionary& dict)
{
    policyPath_ = dict.lookup<fileName>("policy");
    specHash_ = dict.lookup<word>("specHash");
    controlInterval_ = dict.lookup<scalar>("controlInterval");
    startTime_ = dict.lookupOrDefault<scalar>("startTime", 0);
    seed_ = dict.lookupOrDefault<label>("seed", 0);
    deterministic_ = dict.lookupOrDefault<Switch>("deterministic", false);

    hasEndTime_ = dict.found("endTime");
    endTime_ = dict.lookupOrDefault<scalar>("endTime", 0);

    const dictionary& obs = dict.subDict("observation");
    if (obs.lookupOrDefault<label>("stack", 1) != 1)
    {
        FatalErrorInFunction << "stack must be 1" << abort(FatalError);
    }

    PtrList<dictionary> sources(obs.lookup("sources"));
    if (sources.empty())
    {
        FatalErrorInFunction
            << "the observation carries no source" << abort(FatalError);
    }

    sources_.setSize(sources.size());
    forAll(sources, s)
    {
        observationSource& source = sources_[s];
        source.kind = sources[s].lookup<word>("kind");
        source.name = sources[s].lookupOrDefault<word>("name", "probes");

        if (source.kind == "registry")
        {
            continue;
        }
        if (source.kind != "probe")
        {
            FatalErrorInFunction
                << "source " << s << " is of kind " << source.kind
                << ", and this controller reads probe and registry sources"
                << abort(FatalError);
        }

        source.fieldName = sources[s].lookup<word>("fieldName");
        source.interpolation =
            sources[s].lookupOrDefault<word>("interpolation", "cellPoint");
        source.positions = vectorField(sources[s].lookup("positions"));
        source.components =
            sources[s].lookupOrDefault<labelList>("components", labelList());

        forAll(source.components, c)
        {
            if (source.components[c] < 0 || source.components[c] >= vector::nComponents)
            {
                FatalErrorInFunction
                    << "source " << source.name << " asks for component "
                    << source.components[c] << " of " << source.fieldName
                    << ", which has " << label(vector::nComponents)
                    << abort(FatalError);
            }
        }
    }

    if (obsDim() != obs.lookup<label>("dim"))
    {
        FatalErrorInFunction
            << "the observation declares dim " << obs.lookup<label>("dim")
            << " but its " << sources_.size() << " source(s) carry " << obsDim()
            << abort(FatalError);
    }

    const dictionary& act = dict.subDict("action");
    actionName_ = act.lookup<word>("name");
    rampFraction_ = act.lookup<scalar>("rampFraction");

    PtrList<dictionary> targets(act.lookup("targets"));
    targetNames_.setSize(targets.size());
    forAll(targets, i)
    {
        const label component = targets[i].lookup<label>("component");
        if (component < 0 || component >= targets.size())
        {
            FatalErrorInFunction
                << "target " << targets[i].lookup<word>("name")
                << " takes component " << component << " of an action with "
                << targets.size() << " component(s)" << abort(FatalError);
        }
        targetNames_[component] = targets[i].lookup<word>("name");
    }
    low_ = scalarField(act.lookup("low"));
    high_ = scalarField(act.lookup("high"));

    if (low_.size() != high_.size())
    {
        FatalErrorInFunction
            << "the action declares " << low_.size() << " lower bounds and "
            << high_.size() << " upper bounds" << abort(FatalError);
    }
}


void Foam::uqtopusController::locateProbes()
{
    forAll(sources_, s)
    {
        observationSource& source = sources_[s];

        if (source.kind == "registry")
        {
            continue;
        }

        source.cells.setSize(source.positions.size());
        source.owners.setSize(source.positions.size());
        forAll(source.positions, i)
        {
            source.cells[i] = mesh_.findCell(source.positions[i]);
            source.owners[i] =
                source.cells[i] >= 0 ? Pstream::myProcNo() : labelMax;
        }

        // a point on a processor face is found by both neighbours, so the
        // lowest rank owns it and the others stay silent
        Pstream::listCombineGather(source.owners, minEqOp<label>());
        Pstream::listCombineScatter(source.owners);

        forAll(source.positions, i)
        {
            if (source.owners[i] == labelMax)
            {
                FatalErrorInFunction
                    << "probe " << i << " of source " << source.name << " at "
                    << source.positions[i] << " is outside the mesh"
                    << abort(FatalError);
            }
        }
    }
}


Foam::scalarField Foam::uqtopusController::observation() const
{
    scalarField values(obsDim(), 0);

    label offset = 0;
    forAll(sources_, s)
    {
        const observationSource& source = sources_[s];

        if (source.kind == "registry")
        {
            // every rank holds the same value, and the reduction below sums
            if (Pstream::master())
            {
                values[offset] = mesh_.lookupObject<uniformDimensionedScalarField>
                (
                    source.name
                ).value();
            }
        }
        else if (source.components.empty())
        {
            const volScalarField& field =
                mesh_.lookupObject<volScalarField>(source.fieldName);
            const autoPtr<interpolation<scalar>> interpolator
            (
                interpolation<scalar>::New(source.interpolation, field)
            );

            forAll(source.positions, i)
            {
                if (source.owners[i] == Pstream::myProcNo())
                {
                    values[offset + i] = interpolator->interpolate
                    (
                        source.positions[i], source.cells[i]
                    );
                }
            }
        }
        else
        {
            const volVectorField& field =
                mesh_.lookupObject<volVectorField>(source.fieldName);
            const autoPtr<interpolation<vector>> interpolator
            (
                interpolation<vector>::New(source.interpolation, field)
            );
            const label nComponents = source.components.size();

            forAll(source.positions, i)
            {
                if (source.owners[i] == Pstream::myProcNo())
                {
                    const vector sampled(interpolator->interpolate
                    (
                        source.positions[i], source.cells[i]
                    ));
                    forAll(source.components, c)
                    {
                        values[offset + i*nComponents + c] =
                            sampled[source.components[c]];
                    }
                }
            }
        }

        offset += source.size();
    }

    Pstream::listCombineGather(values, plusEqOp<scalar>());
    Pstream::listCombineScatter(values);

    return values;
}


void Foam::uqtopusController::writeRow
(
    const scalarField& observation,
    const scalarField& action
) const
{
    const Time& runTime = mesh_.time();

    if (!trajectory_)
    {
        const fileName dir =
            runTime.rootPath()/runTime.globalCaseName()
           /"postProcessing"/"uqtopusPolicy"
           /runTime.timeName(runTime.startTime().value());
        mkDir(dir);
        trajectory_.reset(new OFstream(dir/"trajectory.dat"));

        OFstream& os = *trajectory_;
        os  << "# uqtopus trajectory" << nl
            << "# specHash         " << specHash_ << nl
            << "# seed             " << seed_ << nl
            << "# columns          time";
        forAll(sources_, s)
        {
            const observationSource& source = sources_[s];

            if (source.kind == "registry")
            {
                os << ' ' << source.name;
                continue;
            }

            forAll(source.positions, i)
            {
                if (source.components.empty())
                {
                    os << ' ' << source.name << '.' << source.fieldName
                       << '.' << i;
                }
                else
                {
                    forAll(source.components, c)
                    {
                        os << ' ' << source.name << '.' << source.fieldName
                           << source.components[c] << '.' << i;
                    }
                }
            }
        }
        if (action.size() == 1)
        {
            os << ' ' << actionName_;
        }
        else
        {
            forAll(action, i)
            {
                os << ' ' << actionName_ << '.' << i;
            }
        }
        os << endl;
        os.precision(10);
    }

    // stamped with the end of the interval this action governs, so that a
    // reward averaged over that interval lines up with it
    OFstream& os = *trajectory_;
    os << startTime_ + nActions_*controlInterval_;
    forAll(observation, i)
    {
        os << ' ' << observation[i];
    }
    forAll(action, i)
    {
        os << ' ' << action[i];
    }
    os << endl;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::uqtopusController::uqtopusController
(
    const fvMesh& mesh,
    const dictionary& dict
)
:
    regIOobject
    (
        IOobject
        (
            registryName,
            mesh.time().constant(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true                    // registered, so the registry owns this
        )
    ),
    mesh_(mesh),
    controlInterval_(0),
    startTime_(0),
    endTime_(0),
    hasEndTime_(false),
    rampFraction_(0),
    low_(),
    high_(),
    seed_(0),
    deterministic_(false),
    curTimeIndex_(-1),
    nActions_(0)
{
    readContract(dict);
    locateProbes();

    policy_.reset(new onnxPolicy(policyPath_, seed_));

    if (policy_->specHash() != specHash_)
    {
        FatalErrorInFunction
            << policyPath_ << " implements contract " << policy_->specHash()
            << " but the case dictionary declares " << specHash_
            << abort(FatalError);
    }
    if (policy_->obsDim() != obsDim())
    {
        FatalErrorInFunction
            << policyPath_ << " expects " << policy_->obsDim()
            << " observations but the dictionary carries " << obsDim()
            << abort(FatalError);
    }

    actionOld_.setSize(policy_->actDim(), 0);
    actionNew_.setSize(policy_->actDim(), 0);
    action_.setSize(policy_->actDim(), 0);

    Info<< "uqtopusController: " << policyPath_ << ", contract "
        << policy_->specHash() << ", " << policy_->obsDim()
        << " observation(s) from " << sources_.size() << " source(s), "
        << policy_->actDim() << " action component(s), every "
        << controlInterval_ << endl;
}


Foam::uqtopusController::~uqtopusController()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

const Foam::dictionary& Foam::uqtopusController::contractDict(const fvMesh& mesh)
{
    const dictionary& controlDict = mesh.time().controlDict();

    if (!controlDict.found(contractName))
    {
        FatalErrorInFunction
            << "controlDict has no " << contractName << " entry. That entry "
            << "carries the policy, the observation and the action, and "
            << "without it nothing can be controlled." << abort(FatalError);
    }

    return controlDict.subDict(contractName);
}


const Foam::uqtopusController& Foam::uqtopusController::New(const fvMesh& mesh)
{
    return New(mesh, contractDict(mesh));
}


Foam::label Foam::uqtopusController::component(const word& target) const
{
    forAll(targetNames_, i)
    {
        if (targetNames_[i] == target)
        {
            return i;
        }
    }
    return -1;
}


const Foam::uqtopusController& Foam::uqtopusController::New
(
    const fvMesh& mesh,
    const dictionary& dict
)
{
    if (!mesh.foundObject<uqtopusController>(registryName))
    {
        // registered in the constructor, so the registry owns it and deletes
        // it with the mesh. Not a leak.
        new uqtopusController(mesh, dict);
    }

    const uqtopusController& controller =
        mesh.lookupObject<uqtopusController>(registryName);

    if (controller.specHash() != dict.lookup<word>("specHash"))
    {
        FatalErrorInFunction
            << "two users of the policy declare different contracts, "
            << controller.specHash() << " and " << dict.lookup<word>("specHash")
            << ". Every target of one action must carry the same block."
            << abort(FatalError);
    }

    return controller;
}


Foam::label Foam::uqtopusController::actDim() const
{
    return action_.size();
}


Foam::label Foam::uqtopusController::obsDim() const
{
    label n = 0;
    forAll(sources_, s)
    {
        n += sources_[s].size();
    }
    return n;
}


const Foam::scalarField& Foam::uqtopusController::action() const
{
    const Time& runTime = mesh_.time();
    const label timeIndex = runTime.timeIndex();

    // every target calls this every time step, and the pressure-velocity
    // coupling calls each of them several times per time step
    if (curTimeIndex_ == timeIndex)
    {
        return action_;
    }

    const scalar t = runTime.value();
    const scalar dt = runTime.deltaTValue();

    const bool started = t >= startTime_ - 0.5*dt;
    const bool finished = hasEndTime_ && t > endTime_ + 0.5*dt;

    // an action decided on the last time step governs an interval the run
    // never reaches, so it is never applied and must not be recorded either
    const bool lastStep = t > runTime.endTime().value() - 0.5*dt;

    if (started && !finished)
    {
        if (!lastStep && t >= startTime_ + nActions_*controlInterval_ - 0.5*dt)
        {
            const scalarField obs(observation());

            // only the master draws, otherwise every process would hold a
            // different action
            if (Pstream::master())
            {
                actionOld_ = actionNew_;
                actionNew_ = policy_->act(obs, deterministic_);
                nActions_++;
                writeRow(obs, actionNew_);
            }
            Pstream::scatter(actionOld_);
            Pstream::scatter(actionNew_);
            Pstream::scatter(nActions_);
        }

        scalar ramp = 1;
        if (rampFraction_ > 0 && nActions_ > 0)
        {
            const scalar since =
                t - (startTime_ + (nActions_ - 1)*controlInterval_);
            ramp = min(max(since/(rampFraction_*controlInterval_), 0), 1);
        }
        action_ = ramp*actionNew_ + (1 - ramp)*actionOld_;

        forAll(action_, i)
        {
            action_[i] = min(max(action_[i], low_[i]), high_[i]);
        }
    }

    curTimeIndex_ = timeIndex;
    return action_;
}


// ************************************************************************* //
