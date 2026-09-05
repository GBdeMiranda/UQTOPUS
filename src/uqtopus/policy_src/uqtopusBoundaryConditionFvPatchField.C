/*---------------------------------------------------------------------------*/

#include "uqtopusBoundaryConditionFvPatchField.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * Helper Functions  * * * * * * * * * * * * * //

template<class Type>
Type Foam::uqtopusDirection(const dictionary& dict)
{
    return dict.lookupOrDefault<Type>("direction", pTraits<Type>::one);
}


template<>
Foam::vector Foam::uqtopusDirection<Foam::vector>(const dictionary& dict)
{
    return dict.lookupOrDefault<vector>("direction", vector(0, 1, 0));
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
    direction_(pTraits<Type>::one),
    rotating_(false),
    origin_(Zero),
    axis_(0, 0, 1),
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
    fixedValueFvPatchField<Type>(p, iF, dict, false),
    dict_(dict),
    component_(-1),
    direction_(uqtopusDirection<Type>(dict)),
    rotating_(dict.found("origin")),
    origin_(dict.lookupOrDefault<vector>("origin", Zero)),
    axis_(dict.lookupOrDefault<vector>("axis", vector(0, 0, 1))),
    controller_(&uqtopusController::New(p.boundaryMesh().mesh()))
{
    // write() puts these two back itself
    dict_.remove("type");
    dict_.remove("value");

    // a patch usually carries the name of the target it drives, so the entry
    // is only needed when the two differ
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

    // no updateCoeffs() here: the field the controller observes is not
    // registered yet while this field is still being constructed
    if (dict.found("value"))
    {
        fvPatchField<Type>::operator=(Field<Type>("value", dict, p.size()));
    }
    else
    {
        fvPatchField<Type>::operator=(pTraits<Type>::zero);
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

    // the controller decides at most once per time step, no matter how many
    // targets ask it or how many times each one asks
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
