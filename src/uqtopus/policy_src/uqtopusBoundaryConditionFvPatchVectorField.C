/*---------------------------------------------------------------------------*/

#include "uqtopusBoundaryConditionFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::uqtopusBoundaryConditionFvPatchVectorField::uqtopusBoundaryConditionFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF),
    dict_(),
    component_(0),
    direction_(0, 1, 0),
    controller_(nullptr)
{}


Foam::uqtopusBoundaryConditionFvPatchVectorField::uqtopusBoundaryConditionFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF, dict, false),
    dict_(dict),
    component_(-1),
    direction_(dict.lookupOrDefault<vector>("direction", vector(0, 1, 0))),
    controller_(&uqtopusController::New(p.boundaryMesh().mesh(), dict))
{
    // write() puts these two back itself
    dict_.remove("type");
    dict_.remove("value");

    PtrList<dictionary> targets(dict.subDict("action").lookup("targets"));
    forAll(targets, i)
    {
        if (targets[i].lookup<word>("name") == p.name())
        {
            component_ = targets[i].lookup<label>("component");
        }
    }

    if (component_ < 0)
    {
        FatalErrorInFunction
            << "no target in the action carries the name of this patch, "
            << p.name() << ". A patch that is not a target has nothing to apply."
            << abort(FatalError);
    }
    if (component_ >= controller_->actDim())
    {
        FatalErrorInFunction
            << "patch " << p.name() << " takes action component " << component_
            << " but the policy returns " << controller_->actDim()
            << " component(s)" << abort(FatalError);
    }

    // no updateCoeffs() here: the field the controller observes is not
    // registered yet while the velocity field is still being constructed
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=(vectorField("value", dict, p.size()));
    }
    else
    {
        fvPatchField<vector>::operator=(vector::zero);
    }
}


Foam::uqtopusBoundaryConditionFvPatchVectorField::uqtopusBoundaryConditionFvPatchVectorField
(
    const uqtopusBoundaryConditionFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(ptf, p, iF, mapper),
    dict_(ptf.dict_),
    component_(ptf.component_),
    direction_(ptf.direction_),
    controller_(ptf.controller_)
{}


Foam::uqtopusBoundaryConditionFvPatchVectorField::uqtopusBoundaryConditionFvPatchVectorField
(
    const uqtopusBoundaryConditionFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(ptf, iF),
    dict_(ptf.dict_),
    component_(ptf.component_),
    direction_(ptf.direction_),
    controller_(ptf.controller_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::uqtopusBoundaryConditionFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // the controller decides at most once per time step, no matter how many
    // targets ask it or how many times each one asks
    vectorField::operator=(direction_*controller_->action()[component_]);

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::uqtopusBoundaryConditionFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    dict_.write(os, false);
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        uqtopusBoundaryConditionFvPatchVectorField
    );
}

// ************************************************************************* //
