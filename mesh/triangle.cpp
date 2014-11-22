// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "mesh_headers.hpp"

const int Triangle::edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};

Triangle::Triangle(const int *ind, int attr) : Element(Geometry::TRIANGLE)
{
   attribute = attr;
   for (int i = 0; i < 3; i++)
      indices[i] = ind[i];
}

Triangle::Triangle(int ind1, int ind2, int ind3, int attr)
   : Element(Geometry::TRIANGLE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
}

int Triangle::NeedRefinement(DSTable &v_to_v, int *middle) const
{
   int m;

   if ((m = v_to_v(indices[0], indices[1])) != -1 && middle[m] != -1) return 1;
   if ((m = v_to_v(indices[1], indices[2])) != -1 && middle[m] != -1) return 1;
   if ((m = v_to_v(indices[2], indices[0])) != -1 && middle[m] != -1) return 1;
   return 0;
}

void Triangle::SetVertices(const int *ind)
{
   for (int i = 0; i < 3; i++)
      indices[i] = ind[i];
}

void Triangle::MarkEdge(DenseMatrix &pmat)
{
   double d[3];
   int shift, v;

   d[0] = ( (pmat(0,1)-pmat(0,0))*(pmat(0,1)-pmat(0,0)) +
            (pmat(1,1)-pmat(1,0))*(pmat(1,1)-pmat(1,0)) );
   d[1] = ( (pmat(0,2)-pmat(0,1))*(pmat(0,2)-pmat(0,1)) +
            (pmat(1,2)-pmat(1,1))*(pmat(1,2)-pmat(1,1)) );
   d[2] = ( (pmat(0,2)-pmat(0,0))*(pmat(0,2)-pmat(0,0)) +
            (pmat(1,2)-pmat(1,0))*(pmat(1,2)-pmat(1,0)) );

   // if pmat has 3 rows, then use extra term in each sum
   if (pmat.Height()==3)
   {
      d[0] += (pmat(2,1)-pmat(2,0))*(pmat(2,1)-pmat(2,0));
      d[1] += (pmat(2,2)-pmat(2,1))*(pmat(2,2)-pmat(2,1));
      d[2] += (pmat(2,2)-pmat(2,0))*(pmat(2,2)-pmat(2,0));
   }

   if (d[0] >= d[1])
      if (d[0] >= d[2]) shift = 0;
      else              shift = 2;
   else
      if (d[1] >= d[2]) shift = 1;
      else              shift = 2;

   switch (shift)
   {
   case 0:
      break;
   case 1:
      v = indices[0];
      indices[0] = indices[1];
      indices[1] = indices[2];
      indices[2] = v;
      break;
   case 2:
      v = indices[0];
      indices[0] = indices[2];
      indices[2] = indices[1];
      indices[1] = v;
      break;
   }
}

void Triangle::MarkEdge(const DSTable &v_to_v, const int *length)
{
   int l, L, j, ind[3], i;

   L = length[ v_to_v(indices[0], indices[1]) ]; j = 0;
   if ( (l = length[ v_to_v(indices[1], indices[2]) ]) > L ) { L = l; j = 1; }
   if ( (l = length[ v_to_v(indices[2], indices[0]) ]) > L ) { L = l; j = 2; }

   for (i = 0; i < 3; i++)
      ind[i] = indices[i];

   switch (j)
   {
   case 1:
      indices[0] = ind[1]; indices[1] = ind[2]; indices[2] = ind[0];
      break;
   case 2:
      indices[0] = ind[2]; indices[1] = ind[0]; indices[2] = ind[1];
      break;
   }
}

void Triangle::GetVertices(Array<int> &v) const
{
   v.SetSize(3);
   for (int i = 0; i < 3; i++)
      v[i] = indices[i];
}

Linear2DFiniteElement TriangleFE;
