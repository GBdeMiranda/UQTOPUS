/*---------------------------------------------------------------------------*\
Class
    Foam::Function1s::uqtopusPolicy

Description
    Scalar Function1 set by one component of the uqtopusController action,
    read from a dictionary passed by reference from its file. Returns value
    until the first decision.

Usage
    \verbatim
    inputA
    {
        type            uqtopusPolicy;
        action          targetA;
        value           0;
    }
    \endverbatim

    \table
        Property | Description                     | Required | Default
        action   | target name in the controller   | yes      |
        value    | value before the first decision | yes      |
    \endtable

SourceFiles
    uqtopusPolicyFunction1.C

\*---------------------------------------------------------------------------*/

#ifndef uqtopusPolicyFunction1_H
#define uqtopusPolicyFunction1_H

#include "Function1.H"
#include "uqtopusController.H"

namespace Foam
{
namespace Function1s
{

class uqtopusPolicy
:
    public FieldFunction1<scalar, uqtopusPolicy>
{
    const uqtopusController& controller_;

    word target_;

    label component_;

    scalar value_;

    static const fvMesh& mesh(const word& name, const dictionary& dict);


public:

    TypeName("uqtopusPolicy");

    uqtopusPolicy(const word& name, const dictionary& dict);

    uqtopusPolicy(const uqtopusPolicy&);

    virtual ~uqtopusPolicy();

    // The action component while a decision is in force, value before
    virtual scalar value(const scalar t) const;

    virtual scalar integral(const scalar t1, const scalar t2) const;

    virtual void write(Ostream& os) const;

    void operator=(const uqtopusPolicy&) = delete;
};


} // End namespace Function1s
} // End namespace Foam

#endif

// ************************************************************************* //
