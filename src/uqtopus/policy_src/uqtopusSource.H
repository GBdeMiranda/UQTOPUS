/*---------------------------------------------------------------------------*\
Class
    Foam::fv::uqtopusSource

Description
    Source term set by one component of the uqtopusController action, added
    to the equation of one field in a set of cells. Adds value until the first
    decision.

Usage
    \verbatim
    heater
    {
        type            uqtopusSource;
        field           T;
        selectionMode   cellZone;
        cellZone        heater;
        volumeMode      specific;
        value           0;
    }
    \endverbatim

    \table
        Property   | Description                       | Required   | Default
        field      | field that receives the source    | yes        |
        volumeMode | specific: per m3, absolute: total | yes        |
        action     | target name in the controller     | no         | entry name
        direction  | multiplies the action             | non-scalar | 1
        value      | source before the first decision  | no         | zero
    \endtable

SourceFiles
    uqtopusSource.C

\*---------------------------------------------------------------------------*/

#ifndef uqtopusSource_H
#define uqtopusSource_H

#include "fvModel.H"
#include "fvCellSet.H"
#include "uqtopusController.H"

namespace Foam
{
namespace fv
{

class uqtopusSource
:
    public fvModel
{
public:

    enum class volumeMode
    {
        absolute,
        specific
    };

    static const NamedEnum<volumeMode, 2> volumeModeNames_;


private:

    fvCellSet set_;

    word fieldName_;

    volumeMode volumeMode_;

    const uqtopusController& controller_;

    label component_;


    void readCoeffs();

    template<class Type>
    void addSupType(fvMatrix<Type>& eqn, const word& fieldName) const;

    template<class Type>
    void addSupType
    (
        const volScalarField&,
        fvMatrix<Type>& eqn,
        const word& fieldName
    ) const;

    template<class Type>
    void addSupType
    (
        const volScalarField&,
        const volScalarField&,
        fvMatrix<Type>& eqn,
        const word& fieldName
    ) const;


public:

    TypeName("uqtopusSource");

    uqtopusSource
    (
        const word& name,
        const word& modelType,
        const dictionary& dict,
        const fvMesh& mesh
    );

    uqtopusSource(const uqtopusSource&) = delete;

    void operator=(const uqtopusSource&) = delete;

    virtual wordList addSupFields() const;

    FOR_ALL_FIELD_TYPES(DEFINE_FV_MODEL_ADD_SUP);

    FOR_ALL_FIELD_TYPES(DEFINE_FV_MODEL_ADD_RHO_SUP);

    FOR_ALL_FIELD_TYPES(DEFINE_FV_MODEL_ADD_ALPHA_RHO_SUP);

    virtual void updateMesh(const mapPolyMesh&);

    virtual bool read(const dictionary& dict);
};


} // End namespace fv
} // End namespace Foam

#endif

// ************************************************************************* //
