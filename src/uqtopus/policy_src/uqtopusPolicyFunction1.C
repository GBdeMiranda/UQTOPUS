/*---------------------------------------------------------------------------*/

#include "uqtopusPolicyFunction1.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace Function1s
{
    makeScalarFunction1(uqtopusPolicy);
}
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const Foam::fvMesh& Foam::Function1s::uqtopusPolicy::mesh
(
    const word& name,
    const dictionary& dict
)
{
    const regIOobject* file = dynamic_cast<const regIOobject*>(&dict.topDict());

    if (!file)
    {
        FatalIOErrorInFunction(dict)
            << name << " is read from " << dict.topDict().name()
            << ", a dictionary copied away from its file" << exit(FatalIOError);
    }

    return file->db().time().lookupObject<fvMesh>(polyMesh::defaultRegion);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::Function1s::uqtopusPolicy::uqtopusPolicy
(
    const word& name,
    const dictionary& dict
)
:
    FieldFunction1<scalar, uqtopusPolicy>(name),
    controller_(uqtopusController::New(mesh(name, dict))),
    target_(dict.lookup<word>("action")),
    component_(controller_.component(target_)),
    value_(dict.lookup<scalar>("value"))
{
    if (component_ < 0 || component_ >= controller_.actDim())
    {
        FatalIOErrorInFunction(dict)
            << "no action component for target " << target_
            << "; the targets are " << controller_.targetNames()
            << exit(FatalIOError);
    }
}


Foam::Function1s::uqtopusPolicy::uqtopusPolicy(const uqtopusPolicy& f)
:
    FieldFunction1<scalar, uqtopusPolicy>(f),
    controller_(f.controller_),
    target_(f.target_),
    component_(f.component_),
    value_(f.value_)
{}


Foam::Function1s::uqtopusPolicy::~uqtopusPolicy()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::scalar Foam::Function1s::uqtopusPolicy::value(const scalar) const
{
    return controller_.active() ? controller_.action()[component_] : value_;
}


Foam::scalar Foam::Function1s::uqtopusPolicy::integral
(
    const scalar,
    const scalar
) const
{
    FatalErrorInFunction
        << name_ << " has no integral" << exit(FatalError);

    return 0;
}


void Foam::Function1s::uqtopusPolicy::write(Ostream& os) const
{
    writeEntry(os, "action", target_);
    writeEntry(os, "value", value_);
}


// ************************************************************************* //
