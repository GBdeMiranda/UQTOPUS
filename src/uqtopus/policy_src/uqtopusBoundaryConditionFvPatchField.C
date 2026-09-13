/*---------------------------------------------------------------------------*/

#include "uqtopusBoundaryConditionFvPatchField.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * Helper Functions  * * * * * * * * * * * * * //

template<class Type>
Type Foam::uqtopusDirection(const dictionary& dict)
{
    return pTraits<Type>::rank == 0
        ? dict.lookupOrDefault<Type>("direction", pTraits<Type>::one)
        : dict.lookup<Type>("direction");
}


template<class Type>
Foam::Field<Type> Foam::uqtopusRotation
(
    const fvPatch& p,
    const scalar,
    const vector&,
    const vector&
)
{
    FatalErrorInFunction
        << "patch " << p.name() << " carries an origin, which makes it a "
        << "rotating wall, and only a vector field can rotate"
        << abort(FatalError);

    return Field<Type>(p.size(), Zero);
}


template<>
Foam::Field<Foam::vector> Foam::uqtopusRotation<Foam::vector>
(
    const fvPatch& p,
    const scalar omega,
    const vector& origin,
    const vector& axis
)
{
    Field<vector> Up(((omega*axis/mag(axis)) ^ (p.Cf() - origin))());
    const vectorField n(p.nf());
    Up -= n*(n & Up);

    return Up;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Type>
Foam::uqtopusBoundaryConditionFvPatchField<Type>::
uqtopusBoundaryConditionFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF
)
:
    fixedValueFvPatchField<Type>(p, iF),
    dict_(),
    component_(0),
    direction_(pTraits<Type>::zero),
    rotating_(false),
    origin_(Zero),
    axis_(Zero),
    controller_(nullptr)
{}


template<class Type>
Foam::uqtopusBoundaryConditionFvPatchField<Type>::
uqtopusBoundaryConditionFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<Type>(p, iF, dict),
    dict_(dict),
    component_(-1),
    direction_
    (
        dict.found("origin")
      ? pTraits<Type>::zero
      : uqtopusDirection<Type>(dict)
    ),
    rotating_(dict.found("origin")),
    origin_(rotating_ ? dict.lookup<vector>("origin") : vector::zero),
    axis_(rotating_ ? dict.lookup<vector>("axis") : vector::zero),
    controller_(&uqtopusController::New(p.boundaryMesh().mesh()))
{
    // write() puts these two back itself
    dict_.remove("type");
    dict_.remove("value");

    const word target(dict.lookupOrDefault<word>("action", p.name()));
    component_ = controller_->component(target);

    if (component_ < 0)
    {
        FatalErrorInFunction
            << "no target of the action is named " << target << ", which is "
            << "what patch " << p.name() << " asks for. The targets are "
            << controller_->targetNames() << abort(FatalError);
    }
    if (component_ >= controller_->actDim())
    {
        FatalErrorInFunction
            << "patch " << p.name() << " takes action component " << component_
            << " but the policy returns " << controller_->actDim()
            << " component(s)" << abort(FatalError);
    }
}


template<class Type>
Foam::uqtopusBoundaryConditionFvPatchField<Type>::
uqtopusBoundaryConditionFvPatchField
(
    const uqtopusBoundaryConditionFvPatchField<Type>& ptf,
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<Type>(ptf, p, iF, mapper),
    dict_(ptf.dict_),
    component_(ptf.component_),
    direction_(ptf.direction_),
    rotating_(ptf.rotating_),
    origin_(ptf.origin_),
    axis_(ptf.axis_),
    controller_(ptf.controller_)
{}


template<class Type>
Foam::uqtopusBoundaryConditionFvPatchField<Type>::
uqtopusBoundaryConditionFvPatchField
(
    const uqtopusBoundaryConditionFvPatchField<Type>& ptf,
    const DimensionedField<Type, volMesh>& iF
)
:
    fixedValueFvPatchField<Type>(ptf, iF),
    dict_(ptf.dict_),
    component_(ptf.component_),
    direction_(ptf.direction_),
    rotating_(ptf.rotating_),
    origin_(ptf.origin_),
    axis_(ptf.axis_),
    controller_(ptf.controller_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Type>
void Foam::uqtopusBoundaryConditionFvPatchField<Type>::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    // the patch keeps its dictionary value until the first decision
    if (controller_->active())
    {
        const scalar a = controller_->action()[component_];

        if (rotating_)
        {
            Field<Type>::operator=
            (
                uqtopusRotation<Type>(this->patch(), a, origin_, axis_)
            );
        }
        else
        {
            Field<Type>::operator=(direction_*a);
        }
    }

    fixedValueFvPatchField<Type>::updateCoeffs();
}


template<class Type>
void Foam::uqtopusBoundaryConditionFvPatchField<Type>::write(Ostream& os) const
{
    fvPatchField<Type>::write(os);
    dict_.write(os, false);
    writeEntry(os, "value", *this);
}


// ************************************************************************* //
