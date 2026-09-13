/*---------------------------------------------------------------------------*\
Class
    Foam::onnxPolicy

Description
    Policy read from an ONNX file and evaluated with ONNX Runtime. The graph
    takes the observation and one standard normal draw per action component,
    and returns the sampled action.

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

class onnxPolicy
{
    // Private data

    fileName path_;

    word specHash_;
    label obsDim_;
    label actDim_;

    Random rng_;

    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;


public:

    onnxPolicy(const fileName& path, const label seed);


    // Member Functions

    const fileName& path() const
    {
        return path_;
    }

    const word& specHash() const
    {
        return specHash_;
    }

    label obsDim() const
    {
        return obsDim_;
    }

    label actDim() const
    {
        return actDim_;
    }

    // One decision, deterministic feeds a zero draw and gives the mean
    tmp<scalarField> act(const scalarField& observation, const bool deterministic);
};


} // End namespace Foam

#endif

// ************************************************************************* //
