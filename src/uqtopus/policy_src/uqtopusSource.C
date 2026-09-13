/*---------------------------------------------------------------------------*/

#include "uqtopusSource.H"
#include "fvMatrices.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace fv
    {
        defineTypeNameAndDebug(uqtopusSource, 0);

        addToRunTimeSelectionTable(fvModel, uqtopusSource, dictionary);
    }

    template<>
    const char* NamedEnum<fv::uqtopusSource::volumeMode, 2>::names[] =
    {
        "absolute",
        "specific"
    };
}

const Foam::NamedEnum<Foam::fv::uqtopusSource::volumeMode, 2>
    Foam::fv::uqtopusSource::volumeModeNames_;


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::fv::uqtopusSource::readCoeffs()
{
    fieldName_ = coeffs().lookup<word>("field");
    volumeMode_ = volumeModeNames_.read(coeffs().lookup("volumeMode"));

    const word target(coeffs().lookupOrDefault<word>("action", name()));
    component_ = controller_.component(target);

    if (component_ < 0 || component_ >= controller_.actDim())
    {
        FatalIOErrorInFunction(coeffs())
            << "no action component for target " << target
            << "; the targets are " << controller_.targetNames()
            << exit(FatalIOError);
    }
}


template<class Type>
void Foam::fv::uqtopusSource::addSupType
(
    fvMatrix<Type>& eqn,
    const word& fieldName
) const
{
    const Type direction
    (
        pTraits<Type>::rank == 0
      ? coeffs().lookupOrDefault<Type>("direction", pTraits<Type>::one)
      : coeffs().lookup<Type>("direction")
    );

    Type source =
        controller_.active()
      ? controller_.action()[component_]*direction
      : coeffs().lookupOrDefault<Type>("value", pTraits<Type>::zero);

    if (volumeMode_ == volumeMode::absolute)
    {
        source /= set_.V();
    }

    typename GeometricField<Type, fvPatchField, volMesh>::Internal Su
    (
        IOobject
        (
            name() + fieldName + "Su",
            mesh().time().timeName(),
            mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh(),
        dimensioned<Type>("zero", eqn.dimensions()/dimVolume, Zero),
        false
    );

    UIndirectList<Type>(Su, set_.cells()) = source;

    eqn += Su;
}


template<class Type>
void Foam::fv::uqtopusSource::addSupType
(
    const volScalarField&,
    fvMatrix<Type>& eqn,
    const word& fieldName
) const
{
    addSupType(eqn, fieldName);
}


template<class Type>
void Foam::fv::uqtopusSource::addSupType
(
    const volScalarField&,
    const volScalarField&,
    fvMatrix<Type>& eqn,
    const word& fieldName
) const
{
    addSupType(eqn, fieldName);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::uqtopusSource::uqtopusSource
(
    const word& name,
    const word& modelType,
    const dictionary& dict,
    const fvMesh& mesh
)
:
    fvModel(name, modelType, dict, mesh),
    set_(coeffs(), mesh),
    fieldName_(),
    volumeMode_(volumeMode::specific),
    controller_(uqtopusController::New(mesh)),
    component_(-1)
{
    readCoeffs();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::wordList Foam::fv::uqtopusSource::addSupFields() const
{
    return wordList(1, fieldName_);
}


FOR_ALL_FIELD_TYPES(IMPLEMENT_FV_MODEL_ADD_SUP, fv::uqtopusSource);

FOR_ALL_FIELD_TYPES(IMPLEMENT_FV_MODEL_ADD_RHO_SUP, fv::uqtopusSource);

FOR_ALL_FIELD_TYPES(IMPLEMENT_FV_MODEL_ADD_ALPHA_RHO_SUP, fv::uqtopusSource);


void Foam::fv::uqtopusSource::updateMesh(const mapPolyMesh& mpm)
{
    set_.updateMesh(mpm);
}


bool Foam::fv::uqtopusSource::read(const dictionary& dict)
{
    if (fvModel::read(dict))
    {
        set_.read(coeffs());
        readCoeffs();
        return true;
    }

    return false;
}


// ************************************************************************* //
