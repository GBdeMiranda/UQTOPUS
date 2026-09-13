/*---------------------------------------------------------------------------*\
Class
    Foam::uqtopusBoundaryConditionFvPatchField

Description
    Fixed value set by one component of the uqtopusController action. Writes
    direction*action on the patch or, when origin is given, the velocity of a
    wall rotating about origin and axis with the action as angular velocity.
    Keeps the value entry until the first decision.

Usage
    \table
        Property  | Description                     | Required    | Default
        action    | target name in the controller   | no          | patch name
        direction | multiplies the action           | non-scalar  | 1
        origin    | point the wall rotates about    | no          |
        axis      | rotation axis                   | with origin |
        value     | value before the first decision | yes         |
    \endtable

SourceFiles
    uqtopusBoundaryConditionFvPatchField.C

\*---------------------------------------------------------------------------*/

#ifndef uqtopusBoundaryConditionFvPatchField_H
#define uqtopusBoundaryConditionFvPatchField_H

#include "fixedValueFvPatchFields.H"
#include "uqtopusController.H"

namespace Foam
{

template<class Type>
Type uqtopusDirection(const dictionary&);

template<class Type>
Field<Type> uqtopusRotation
(
    const fvPatch&,
    const scalar,
    const vector&,
    const vector&
);

template<class Type>
class uqtopusBoundaryConditionFvPatchField:
    public fixedValueFvPatchField<Type>
{
    dictionary dict_;

    label component_;

    Type direction_;

    bool rotating_;

    vector origin_;
    vector axis_;

    const uqtopusController* controller_;


public:

    TypeName("uqtopusBoundaryCondition");

    uqtopusBoundaryConditionFvPatchField
    (
        const fvPatch&,
        const DimensionedField<Type, volMesh>&
    );

    uqtopusBoundaryConditionFvPatchField
    (
        const fvPatch&,
        const DimensionedField<Type, volMesh>&,
        const dictionary&
    );

    uqtopusBoundaryConditionFvPatchField
    (
        const uqtopusBoundaryConditionFvPatchField<Type>&,
        const fvPatch&,
        const DimensionedField<Type, volMesh>&,
        const fvPatchFieldMapper&
    );

    uqtopusBoundaryConditionFvPatchField
    (
        const uqtopusBoundaryConditionFvPatchField<Type>&
    ) = delete;

    uqtopusBoundaryConditionFvPatchField
    (
        const uqtopusBoundaryConditionFvPatchField<Type>&,
        const DimensionedField<Type, volMesh>&
    );

    virtual tmp<fvPatchField<Type>> clone
    (
        const DimensionedField<Type, volMesh>& iF
    ) const
    {
        return tmp<fvPatchField<Type>>
        (
            new uqtopusBoundaryConditionFvPatchField<Type>(*this, iF)
        );
    }


    // Member Functions

    virtual void updateCoeffs();

    virtual void write(Ostream&) const;
};


} // End namespace Foam

#ifdef NoRepository
    #include "uqtopusBoundaryConditionFvPatchField.C"
#endif

#endif

// ************************************************************************* //
