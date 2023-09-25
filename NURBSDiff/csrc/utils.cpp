#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define eps 1.0e-4
#define pmax 10

// Algorithm A2.1 (page 68)
int find_span(int n, int p, float u, float* U)
{
      //if (u == U[n+1])
      //      return n; // Special case
      if (fabs(u - U[n+1]) < eps)
            return n - 1; // Special case
      int low  = p;
      int high = n+1; 
      // Do binary search
      int mid = (low + high)/2;
      while (u < U[mid]-eps || u >= U[mid+1]-eps){
            if (u < U[mid]-eps)
                  high = mid;
            else
                  low = mid;
            mid = (low + high)/2;
      }
      return mid;
}


// Algorithm A2.2 (page 70)
void basis_funs(int i, float u, int p, float* U, float* N)
{
      float *left  = new float [p+1];
      float *right = new float [p+1];
      float saved, temp;
      N[0] = 1.0;
      for (int j = 1; j <= p; j += 1){
            left[j] = u-U[i+1-j];
            right[j] = U[i+j]-u;
            saved = 0.0;
            for (int r = 0; r < j; r += 1){
                  temp = N[r]/(right[r+1] + left[j-r]);
                  N[r] = saved + right[r+1]*temp;
                  saved = left[j-r]*temp;
            }
            N[j] = saved;
      }
      
      delete[] left;
      delete[] right;
}
