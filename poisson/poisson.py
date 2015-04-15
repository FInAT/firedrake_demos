from firedrake import *
import finat
import FIAT

mesh = UnitSquareMesh(4, 4, quadrilateral=True)

fs = FunctionSpace(mesh, "CG", 3)

f = Function(fs)
f.interpolate(Expression("sin(2*pi*x[0])*sin(2*pi*x[1])"))

u = Function(fs)
v = TestFunction(fs)

solve(inner(grad(u), grad(v))*dx - v*f*dx == 0, u)

#File("poisson.vtu") << u

fe = finat.ufl_interface.element_from_ufl(fs._ufl_element)

q = finat.quadrature.GaussLobattoQuadrature(fe.cell, 4)
qi = finat.TensorPointIndex(q.points)

x_el = finat.VectorFiniteElement(fe, 2)
x = finat.Variable("x")

kernel_data = finat.KernelData(x_el, x)

form = fe.moment_evaluation(fe.basis_evaluation(qi, kernel_data), q.weights, qi, kernel_data)

form = finat.GeometryMapper(kernel_data)(form)
form = finat.mappers.FactorDeltaMapper()(form)
form = finat.mappers.CancelDeltaMapper()(form)
form = finat.mappers.BindingMapper(kernel_data)(form)

f = finat.Variable("f")
pform = fe.moment_evaluation(fe.field_evaluation(f, qi, kernel_data, finat.grad), q.weights, qi, kernel_data, finat.grad)
pform = finat.GeometryMapper(kernel_data)(pform)
pform = finat.mappers.CancelCompoundVectorMapper()(pform)
print pform
pform = finat.mappers.FactorDeltaMapper()(pform)
pform = finat.mappers.CancelDeltaMapper()(pform)
pform = finat.mappers.IndexSumMapper(kernel_data)(pform)
print pform
