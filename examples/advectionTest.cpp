//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

int problem;
double omega;

// Exact solution
double u_exact(const double x, const double y, const double t);

// Exact derivates
double u_t(const double x, const double y, const double t);
double u_x(const double x, const double y, const double t);
double u_y(const double x, const double y, const double t);

// Prescribed time-independent boundary and right-hand side functions.
void bdr_func(const Vector &X, Vector &V);
double rhs_func(const Vector &X, double t);

// Velocity coefficient (returns a = [1,1])
void velocity_function(const Vector &x, Vector &v);

class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // const char *mesh_file = "../data/unitGridTestMesh.msh";
   // const char *mesh_file = "../data/squareBoundaryMesh.msh";
   const char *mesh_file = "../data/square.msh";
   // const char *mesh_file = "../data/singleelem.msh";
   // const char *mesh_file = "../data/twoelem.msh";
   // const char *mesh_file = "../data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool sbp = 0;
	problem = 1;
	omega = 1;
	int ref_levels = 0;
	int N = 3; // number of time levels -- add option to be parsed
   double tau_final = 0.5;
   double dtau = 0.001;
   bool pseudotime = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&sbp, "-sbp", "--summationbyparts", "-no-sbp",
                  "--no-summationbyparts",
                  "Enable or disable use of SBP operators.");
	args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use: 0 = linear displacement, "
						"2 == quadratic displacement.");
	args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of initial uniform refinement levels.");
   args.AddOption(&N, "-n", "--time-levels",
                  "Numbers of time levels to use. Default N = 1.");
   args.AddOption(&omega, "-f", "--frequency",
                  "Exact solution frequency");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   ODESolver *ode_solver = new RK4Solver();

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (sbp)
   {
      fec = new H1_SBPCollection(order, dim);
   }
   else if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      mfem::out << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, N, Ordering::byNODES);
   mfem::out << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

	/// MOVED ABOVE 
   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 100.0;

   VectorFunctionCoefficient bdr(N, bdr_func);

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   mfem::out << "mesh boundary attributes: \n";
   mesh->bdr_attributes.Print();
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      
      // Mark boundary attributes 1 and 4 as Dirichlet 
      ess_bdr = 0;
      ess_bdr[0] = 1;
      ess_bdr[3] = 1;
      mfem::out << "ess_bdr: \n";
      ess_bdr.Print(mfem::out, 100);

		// Project boundary conditions onto grid function to strongly impose
		// boundary conditions. BC's are defined in the function `bdr_func`.
		x.ProjectBdrCoefficient(bdr, ess_bdr);
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   VectorFunctionCoefficient velocity(dim, velocity_function);
   // FunctionCoefficient inflow(inflow_function);
   // FunctionCoefficient u0(u0_function);
   ConstantCoefficient one(1.0);

	// construct bilinear form associated with spatial discretization
   BilinearForm *k = new BilinearForm(fespace);

   // add domain integrator associated with time spectral operator
   k->AddDomainIntegrator(new TimeSpectralOperatorIntegrator(one, N, omega));

   // add domain integrator assosciated with advection discretization
   k->AddDomainIntegrator(new TimeSpectralConvectionIntegrator(velocity, N, 1.0));
   // k->AddDomainIntegrator(new ConvectionIntegrator(velocity, 1.0));

   // create linear form
   LinearForm *b = new LinearForm(fespace);

   // Functio coefficient for exact rhs function for forcing
   FunctionCoefficient rhs(rhs_func);

   // Should be:
   // LinearFormIntegrator *lfi = new DomainLFIntegrator(rhs);
   // DomainLFIntegrator *lfi = new DomainLFIntegrator(rhs);
	b->AddDomainIntegrator(new TSLFIntegrator(new DomainLFIntegrator(rhs), N, omega));

   int skip_zeros = 0; // 0?
   mfem::out << "above assemble\n";
   k->Assemble(skip_zeros);
   mfem::out << "below\n";
   k->Finalize(skip_zeros);
   b->Assemble();

   GridFunction g(fespace);

   if (pseudotime)
   {
      // Construct bilinear form associated with pseudo-time stepping
      BilinearForm *m = new BilinearForm(fespace);
      m->AddDomainIntegrator(new TSMassIntegrator(one, N));
      // mfem::out << "above m assemble\n";
      m->Assemble();
      // mfem::out << "below pseudotime\n";
      m->Finalize();

      // Project exact solution as initial condition
      ConstantCoefficient zero(1.0);
      x.ProjectCoefficient(bdr);
      
      // 8. Define the time-dependent evolution operator describing the ODE
      //    right-hand side, and perform time-integration (looping over the time
      //    iterations, ti, with a time-step dt).
      FE_Evolution adv(m->SpMat(), k->SpMat(), *b);

      double tau = 0.0;
      adv.SetTime(tau);
      ode_solver->Init(adv);

      bool done = false;
      for (int ti = 0; !done; )
      {
         double dtau_real = min(dtau, tau_final - tau);
         ode_solver->Step(x, tau, dtau_real);
         ti++;

         done = (tau >= tau_final - 1e-8*dtau);

         if (done || ti % 5 == 0)
         {
            mfem::out << "time step: " << ti << ", time: " << tau << endl;
         }
      }
   }
   else
   {
      char outfileName[32];
		snprintf(outfileName, 32, "matrixsystem.txt");
      ofstream outputFile;
      outputFile.open(outfileName, ios::out); 
      int rowdata[] = {0,1,2,3,4,5,6,7,8};
      Array<int> rows(*rowdata);

      DenseMatrix subm;
      subm.SetSize(9);
      subm = 0.0;
      if (outputFile.is_open())
      {
         outputFile << "k = [";
         k->SpMat().PrintMatlab(outputFile);
         outputFile << "];\n";
         // k->SpMat().GetSubMatrix(rows, rows, subm);
         // mfem::out << "A1: \n";
         // subm.PrintMatlab();
         SparseMatrix A;
         Vector B, X;
         k->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
         mfem::out << "A2: \n";
         k->SpMat().PrintInfo(mfem::out);
         outputFile << "k2 = [";
         k->SpMat().PrintMatlab(outputFile);
         outputFile << "];\n";
         // mfem::out << "X: \n";
         // X.Print_HYPRE(mfem::out);
         // mfem::out << "B: \n";
         // B.Print_HYPRE(mfem::out);

         cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
         // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
         //     solve the system A X = B with PCG.
         GSSmoother M(A);
         PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
         // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(A);
         umf_solver.Mult(B, X);
#endif
         outputFile << "X = [";
         X.Print_HYPRE(outputFile);
         outputFile << "];\n";

         // 11. Recover the solution as a finite element grid function.
         k->RecoverFEMSolution(X, *b, x);

         outputFile << "x = [";
         x.Print_HYPRE(outputFile);
         outputFile << "];\n";
         outputFile << "b = [";
         b->Print_HYPRE(outputFile);
         outputFile << "];\n";

         // 11b. Create grid function g and project solution onto it for visualizetion
         GridFunction g(fespace);
         g = 0.0;
         g.ProjectCoefficient(bdr);

         outputFile << "g =[";
         g.Print_HYPRE(outputFile);
         outputFile << "];\n";
      }
      outputFile.close();
   }

   // // 11b. Create grid function g and project solution onto it for visualizetion
   // GridFunction g(fespace);
   g = 0.0;
   g.ProjectCoefficient(bdr);

   // GridFunction err(fespace);
   // err = 0.0;
   // err = x-g;

   // mfem::out << "g:\n";
   // g.Print_HYPRE(mfem::out);

   // mfem::out << "g"
   mfem::out << "\ng err:\n|| u_h - u ||_{L^2} = " << g.ComputeL2Error(bdr) << '\n' << endl;

   ofstream outg("exactSol.vtk");
   outg.precision(14); 
   mesh->PrintVTK(outg, order); 
   g.SaveVTK(outg, "sol", order);

	// 12. Compute and print the L^2 norm of the error.
   mfem::out << "x.vdim: " << x.VectorDim() << " x.size: " << x.Size() << "\n";
   mfem::out << "\n|| u_h - u ||_{L^2} = " << x.ComputeL2Error(bdr) << '\n' << endl;
	mfem::out << "h: " << 0.1 / pow(2, ref_levels) << "\n";

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   // ofstream mesh_ofs("refined.mesh");
   // mesh_ofs.precision(8);
   // mesh->Print(mesh_ofs);
   // mesh->PrintVTK(mesh_ofs); 
   // ofstream sol_ofs("sol.gf");
   // sol_ofs.precision(8);
   // x.Save(sol_ofs);

   char solFileName[32];

   snprintf(solFileName, 32, "adv_TS_P%d_N%d.vtk", order, N);

   ofstream omesh(solFileName);
   omesh.precision(14); 
   mesh->PrintVTK(omesh, order); 
   x.SaveVTK(omesh, "sol", order);

   // ofstream outErr("err.vtk");
   // outErr.precision(14); 
   // mesh->PrintVTK(outErr, order); 
   // err.SaveVTK(outErr, "err", order);

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 14. Free the used memory.
   // delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   _K *= -1.0;
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void velocity_function(const Vector &x, Vector &v)
{
   v(0) = 1.0;
   v(1) = 1.0;
}

// Used for the Dirichlet BC at each time level
// This function works correctly
void bdr_func(const Vector &X, Vector &V)
{
	double x = X(0), y = X(1);
	const int N = V.Size();
	const double T = 2*M_PI/omega;
	const double dt = T / (N);
	for (int i = 0; i < N; i++)
	{
		double t = dt * (i);
		V(i) = u_exact(x, y, t);
	}
}

// exact time domain solution
double u_exact(const double x, const double y, const double t)
{
	double u = std::cos(0.5*x + 0.5*y - t);
	// double u = std::cos(x + y - t);
	return u;
}

// exact derivates
double u_t(const double x, const double y, const double t)
{
   double ut = omega * std::sin(x + y - omega*t);
   return ut;
}
double u_x(const double x, const double y, const double t)
{
   double ux =  -std::sin(x + y - omega*t);
   return ux;
}
double u_y(const double x, const double y, const double t)
{
   double uy = -std::sin(x + y - omega*t);
   return uy;
}

// right hand side function for manufactured solution
double rhs_func(const Vector &X, double t)
{
	double x = X(0), y = X(1), z = 0.0;

   z = 0.0;
   // z = -sin(x + y - t);
   return z;
}
