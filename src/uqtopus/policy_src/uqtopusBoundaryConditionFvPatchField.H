/*---------------------------------------------------------------------------*\
Class
    Foam::uqtopusBoundaryConditionFvPatchField

Description
    Boundary condition driven by one component of the action a uqtopusController
    decides. Serves every field type: the action is one scalar either way, and
    what the patch does with it is what differs.

    Holds no policy of its own. It finds the controller in the mesh registry,
    creating it on first use, and every time step writes the action on the faces
    of its patch. Which component it takes comes from the target entry that
    carries this patch name.

    Without an origin every face gets

        direction * action[component]

    where direction carries the type of the field: a gain on a scalar field, a
    direction on a vector one.

    With an origin the patch is a wall rotating about the axis through it, and
    the face at c gets

        (action[component] * axis) ^ (c - origin)

    with the component normal to the wall removed. The action is then an angular
    velocity, and the bounds in the spec are in rad/s. Only a vector field can
    rotate, so an origin on any other field type is an error.

    The dictionary entries are the ones rendered by
    uqtopus.rl.render_controller(), plus three optional ones:

    \table
        Property  | Description                            | Default
        direction | what the action is multiplied by       | 1, or (0 1 0)
        origin    | point the patch rotates about          | none, no rotation
        axis      | axis the patch rotates about           | (0 0 1)
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

//- What the action is multiplied by when the patch does not rotate. A vector
//  field keeps the jet direction it always had; anything else takes a gain.
template<class Type>
Type uqtopusDirection(const dictionary&);

//- Velocity of a patch rotating about an axis, defined for vector fields only
template<class Type>
Field<Type> uqtopusRotation
(
    const fvPatch&,
    const scalar,
    const vector&,
    const vector&
);


/*---------------------------------------------------------------------------*\
              Class uqtopusBoundaryConditionFvPatchField Declaration
\*---------------------------------------------------------------------------*/

template<class Type>
class uqtopusBoundaryConditionFvPatchField
:
    public fixedValueFvPatchField<Type>
{
    // Private data

        //- The dictionary as read, minus type and value, for write()
        dictionary dict_;

        //- Which component of the action this patch applies
        label component_;

        //- What the action is multiplied by, when the patch does not rotate
        Type direction_;

        //- Whether the patch rotates instead of translating
        bool rotating_;

        //- Point and axis the patch rotates about
        vector origin_;
        vector axis_;

        //- Owned by the mesh registry, shared with every other target
        const uqtopusController* controller_;


public:

    //- Runtime type information
    TypeName("uqtopusBoundaryCondition");


    // Constructors

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

        //- Disallow copy without setting internal field reference
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

        //- Take this patch's component of the action and write it
        virtual void updateCoeffs();

        //- Write, so that a written time directory can be read back
        virtual void write(Ostream&) const;
};


} // End namespace Foam

#ifdef NoRepository
    #include "uqtopusBoundaryConditionFvPatchField.C"
#endif

#endif

// ************************************************************************* //
