//                       MFEM Example 11 - Parallel Version
//
// Compile with: make ex11p
//
// Sample runs:  mpirun -np 4 ex11p -m ../data/square-disc.mesh
//               mpirun -np 4 ex11p -m ../data/star.mesh
//               mpirun -np 4 ex11p -m ../data/escher.mesh
//               mpirun -np 4 ex11p -m ../data/fichera.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex11p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex11p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex11p -m ../data/star-surf.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex11p -m ../data/inline-segment.mesh
//
// Description:  This example code demonstrates the use of MFEM to solve a
//               generalized eigenvalue problem
//                 -Delta u = lambda u
//               with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize the Laplacian operator using a
//               FE space of the specified order, or if order < 1 using an
//               isoparametric/isogeometric space (i.e. quadratic for
//               quadratic curvilinear mesh, NURBS for NURBS mesh, etc.)
//
//               The example is a modification of example 1 which highlights
//               the use of the LOBPCG eigenvalue solver in HYPRE.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int nev = 5;
   int sr = 2, pr = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&sr, "-sr", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 8. Set up the parallel bilinear forms a(.,.) and m(.,.) on the
   //    finite element space.  The first corresponds to the Laplacian
   //    operator -Delta, by adding the Diffusion domain integrator and
   //    imposing homogeneous Dirichlet boundary conditions. The boundary
   //    conditions are implemented by marking all the boundary attributes
   //    from the mesh as essential.  The second is a simple mass matrix
   //    needed on the right hand side of the generalized eigenvalue problem.
   //    After serial and parallel assembly we extract the corresponding
   //    parallel matrix A.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   m->Finalize();

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();

   delete a;
   delete m;

   // 9. Define a parallel grid function to approximate each of the
   // eigenmodes returned by the solver.  Use this as a template to
   // create a special multi-vector object needed by the eigensolver
   // which is then initialized with random values.
   ParGridFunction x(fespace);
   x = 0.0;

   HypreParVector   * X = x.ParallelAverage();
   HypreMultiVector * eigenvectors = new HypreMultiVector(nev, *X);

   int seed = 123;
   eigenvectors->Randomize(seed);

   // 10. Define and configure the LOBPCG eigensolver and a BoomerAMG
   //     preconditioner to be used within the solver.
   HypreLOBPCG * lobpcg = new HypreLOBPCG(eigenvectors->GetInterpreter());
   HypreSolver *    amg = new HypreBoomerAMG(*A);

   lobpcg->SetPrecond(*amg);
   lobpcg->SetMaxIter(100);
   lobpcg->SetTol(1e-6);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);

   // Set the matrices which define the linear system
   lobpcg->SetupB(*M, *X);
   lobpcg->Setup(*A, *X, *X);

   // Obtain the eigenvalues and eigenvectors
   Vector eigenvalues(nev);
   eigenvalues = -1.0;

   lobpcg->Solve(eigenvalues, *eigenvectors);

   // 11. Save the refined mesh and the modes in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         x = eigenvectors->GetVector(i);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");
      }
   }

   // 12. Send the solution by socket to a GLVis server.

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         x = eigenvectors->GetVector(i);

         mode_sock << "parallel " << num_procs << " " << myid << "\n";
         mode_sock << "solution\n" << *pmesh << x << flush;

         char c;
         if (myid == 0)
         {
            cout << "press (q)uit or (c)ontinue --> " << flush;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }
      mode_sock.close();
   }

   // 13. Free the used memory.
   delete amg;
   delete lobpcg;
   delete eigenvectors;
   delete X;
   delete M;
   delete A;

   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
