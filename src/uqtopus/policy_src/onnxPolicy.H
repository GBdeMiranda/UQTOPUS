/*---------------------------------------------------------------------------*\
Class
    Foam::onnxPolicy

Description
    A policy network read from an ONNX file, evaluated with ONNX Runtime.

    Holds no OpenFOAM object beyond basic types, so the same class serves a
    boundary condition, a source term or a well model. It answers one question:
    given an observation, what is the action.

    The graph takes the observation and one standard normal draw per action
    component, and returns the action, already sampled and already clipped to
    the action bounds. The draw is
    made here rather than inside the graph because an ONNX random operator takes
    its seed as a graph attribute, and every episode of one training iteration
    shares a single policy file.

    Implements uqtopus contract 2.0.

SourceFiles
    onnxPolicy.C

\*---------------------------------------------------------------------------*/

#ifndef onnxPolicy_H
#define onnxPolicy_H

#include <memory>

#include <onnxruntime_cxx_api.h>

#include "fileName.H"
#include "scalarField.H"
#include "Random.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class onnxPolicy Declaration
\*---------------------------------------------------------------------------*/

class onnxPolicy
{
    // Private data

        //- The file this policy was read from
        fileName path_;

        //- Contract hash and the two widths, read from the file metadata
        word specHash_;
        label obsDim_;
        label actDim_;

        //- Draws the noise the graph is fed
        Random rng_;

        //- ONNX Runtime state, shared by every copy
        std::shared_ptr<Ort::Env> env_;
        std::shared_ptr<Ort::Session> session_;


public:

    // Constructors

        //- Open an ONNX file and check that it implements the contract
        onnxPolicy(const fileName& path, const label seed);


    // Member Functions

        const fileName& path() const
        {
            return path_;
        }

        //- Contract hash, to be compared with the one in the case dictionary
        const word& specHash() const
        {
            return specHash_;
        }

        //- Number of observation components the graph expects
        label obsDim() const
        {
            return obsDim_;
        }

        //- Number of action components the graph returns
        label actDim() const
        {
            return actDim_;
        }

        //- One decision. deterministic passes a zero draw, giving the mean.
        tmp<scalarField> act(const scalarField& observation, const bool deterministic);
};


} // End namespace Foam

#endif

// ************************************************************************* //
