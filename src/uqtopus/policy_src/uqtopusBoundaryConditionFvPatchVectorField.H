/*---------------------------------------------------------------------------*\
Class
    Foam::uqtopusBoundaryConditionFvPatchVectorField

Description
    Velocity boundary condition driven by one component of the action a
    uqtopusController decides.

    Holds no policy of its own. It finds the controller in the mesh registry,
    creating it on first use, and every time step writes

        direction * action[component]

    on the faces of its patch. Which component it takes comes from the target
    entry that carries this patch name.

    The dictionary entries are the ones rendered by
    uqtopus.rl.render_controller(), plus one optional entry:

    \table
        Property  | Description                            | Default
        direction | direction the action is applied along  | (0 1 0)
    \endtable

SourceFiles
    uqtopusBoundaryConditionFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef uqtopusBoundaryConditionFvPatchVectorField_H
#define uqtopusBoundaryConditionFvPatchVectorField_H

#include "fixedValueFvPatchFields.H"
#include "uqtopusController.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class uqtopusBoundaryConditionFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class uqtopusBoundaryConditionFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- The dictionary as read, minus type and value, for write()
        dictionary dict_;

        //- Which component of the action this patch applies
        label component_;

        //- Direction the action is applied along
        vector direction_;

        //- Owned by the mesh registry, shared with every other target
        const uqtopusController* controller_;


public:

    //- Runtime type information
    TypeName("uqtopusBoundaryCondition");


    // Constructors

        uqtopusBoundaryConditionFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        uqtopusBoundaryConditionFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        uqtopusBoundaryConditionFvPatchVectorField
        (
            const uqtopusBoundaryConditionFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Disallow copy without setting internal field reference
        uqtopusBoundaryConditionFvPatchVectorField
        (
            const uqtopusBoundaryConditionFvPatchVectorField&
        ) = delete;

        uqtopusBoundaryConditionFvPatchVectorField
        (
            const uqtopusBoundaryConditionFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new uqtopusBoundaryConditionFvPatchVectorField(*this, iF)
            );
        }


    // Member Functions

        //- Take this patch's component of the action and write it
        virtual void updateCoeffs();

        //- Write, so that a written time directory can be read back
        virtual void write(Ostream&) const;
};


} // End namespace Foam

#endif

// ************************************************************************* //
