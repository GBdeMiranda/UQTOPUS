/*---------------------------------------------------------------------------*\
Class
    Foam::uqtopusController

Description
    One policy decision per control step, shared through the mesh registry by
    every target of the action. Owns the policy, the probes, the schedule, the
    ramp, the bounds and the trajectory file.

SourceFiles
    uqtopusController.C

\*---------------------------------------------------------------------------*/

#ifndef uqtopusController_H
#define uqtopusController_H

#include <memory>

#include "regIOobject.H"
#include "fvMesh.H"
#include "OFstream.H"
#include "onnxPolicy.H"

namespace Foam
{

class uqtopusController
:
    public regIOobject
{
    // Private data

    const fvMesh& mesh_;

    // A probe source reads a field at points, scalar when components is
    // empty. A registry source reads one scalar published under its name.
    struct observationSource
    {
        word kind;
        word name;

        word fieldName;
        word interpolation;
        labelList components;
        vectorField positions;

        labelList cells;
        labelList owners;

        label size() const
        {
            return kind == "registry"
                ? 1
                : positions.size()*max(components.size(), label(1));
        }
    };

    fileName policyPath_;
    word specHash_;
    word actionName_;
    scalar controlInterval_;
    scalar startTime_;
    scalar endTime_;
    bool hasEndTime_;
    scalar rampFraction_;
    scalarField low_;
    scalarField high_;
    label seed_;
    bool deterministic_;

    List<observationSource> sources_;

    wordList targetNames_;

    mutable scalarField actionOld_;
    mutable scalarField actionNew_;
    mutable scalarField action_;
    mutable label curTimeIndex_;
    mutable label nActions_;

    mutable std::unique_ptr<onnxPolicy> policy_;
    mutable std::unique_ptr<OFstream> trajectory_;


    // Private Member Functions

    void readContract(const dictionary&);

    void locateProbes();

    scalarField observation() const;

    void writeRow(const scalarField&, const scalarField&) const;

    void update() const;


public:

    static const word registryName;

    static const word contractName;

    TypeName("uqtopusController");

    uqtopusController(const fvMesh&, const dictionary&);

    uqtopusController(const uqtopusController&) = delete;

    void operator=(const uqtopusController&) = delete;

    virtual ~uqtopusController();


    // Member Functions

    static const uqtopusController& New(const fvMesh&, const dictionary&);

    static const uqtopusController& New(const fvMesh&);

    static const dictionary& contractDict(const fvMesh&);

    label component(const word& target) const;

    const wordList& targetNames() const
    {
        return targetNames_;
    }

    const word& specHash() const
    {
        return specHash_;
    }

    label actDim() const;

    label obsDim() const;

    // Whether a decision is in force
    bool active() const;

    // The ramped and clipped action, an error while not active
    const scalarField& action() const;

    virtual bool writeData(Ostream&) const
    {
        return true;
    }
};


} // End namespace Foam

#endif

// ************************************************************************* //
