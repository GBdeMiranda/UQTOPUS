/*---------------------------------------------------------------------------*/

#include "uqtopusBoundaryConditionFvPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"

// One instantiation per field type, so a scalar field such as p and a vector
// field such as U carry the same boundary condition.

namespace Foam
{
    makePatchFields(uqtopusBoundaryCondition);
}

// ************************************************************************* //
