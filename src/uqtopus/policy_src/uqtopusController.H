/*---------------------------------------------------------------------------*\
Class
    Foam::uqtopusController

Description
    The one decision per control step, shared by every target it drives.

    Stored in the mesh registry, so that N boundary conditions, source terms or
    any other user of the same policy see the same action vector. Without a
    single shared instance each of them would draw its own noise, run the graph
    on its own, and write over the others' trajectory file.

    Owns the policy, the probe positions, the schedule, the ramp and the
    trajectory file. Users of the action own only the decision of what to do
    with their component of it.

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

/*---------------------------------------------------------------------------*\
                     Class uqtopusController Declaration
\*---------------------------------------------------------------------------*/

class uqtopusController
:
    public regIOobject
{
    // Private data

        const fvMesh& mesh_;

        //- One source of observation, of one of two kinds. A probe reads a
        //  field at points, and an empty components list means that field is
        //  scalar. A registry source reads one scalar the case published under
        //  a name.
        struct observationSource
        {
            word kind;
            word name;

            //- Probe only
            word fieldName;
            word interpolation;
            labelList components;
            vectorField positions;

            //- Cell holding each probe, and the process that owns it
            labelList cells;
            labelList owners;

            //- How many numbers this source contributes
            label size() const
            {
                return kind == "registry"
                    ? 1
                    : positions.size()*max(components.size(), label(1));
            }
        };

        //- Everything read from the dictionary
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

        //- Concatenated in declaration order to form the observation
        List<observationSource> sources_;

        //- Target names, in action component order
        wordList targetNames_;

        //- Set once per control step, then held
        mutable scalarField actionOld_;
        mutable scalarField actionNew_;

        //- Ramped between the two, recomputed every time step
        mutable scalarField action_;

        //- Guard against deciding more than once per time step
        mutable label curTimeIndex_;

        //- Decisions taken so far
        mutable label nActions_;

        mutable std::unique_ptr<onnxPolicy> policy_;
        mutable std::unique_ptr<OFstream> trajectory_;


    // Private Member Functions

        void readContract(const dictionary&);
        void locateProbes();
        scalarField observation() const;
        void writeRow(const scalarField&, const scalarField&) const;


public:

    //- Name this object is registered under
    static const word registryName;

    //- Name of the controlDict entry that carries the contract
    static const word contractName;

    TypeName("uqtopusController");


    // Constructors

        uqtopusController(const fvMesh&, const dictionary&);

        //- Disallow copying: there is one per case, in the registry
        uqtopusController(const uqtopusController&) = delete;
        void operator=(const uqtopusController&) = delete;


    //- Destructor
    virtual ~uqtopusController();


    // Member Functions

        //- Find the controller in the registry, creating it on first use
        static const uqtopusController& New(const fvMesh&, const dictionary&);

        //- Find the controller in the registry, creating it from the contract
        //  that controlDict carries
        static const uqtopusController& New(const fvMesh&);

        //- The contract block, the contractName entry of controlDict
        static const dictionary& contractDict(const fvMesh&);

        //- Which component of the action drives the named target, -1 if none
        label component(const word& target) const;

        //- Target names, in action component order
        const wordList& targetNames() const
        {
            return targetNames_;
        }

        const word& specHash() const
        {
            return specHash_;
        }

        label actDim() const;

        //- Length of the observation, over every source
        label obsDim() const;

        //- Decide if a control step is due, then return the ramped action.
        //  Safe to call many times per time step and from many users.
        const scalarField& action() const;

        //- Nothing of this belongs in a time directory
        virtual bool writeData(Ostream&) const
        {
            return true;
        }
};


} // End namespace Foam

#endif

// ************************************************************************* //
