#include <iostream>
#include <stdlib.h> 
#include <mpi.h>
#include <time.h>
#include <cmath>

using namespace std;

static const double PI = 3.1415926536; 

int main(int argc, char* argv[]){

  int rank, size, ierr;
  MPI_Comm comm;
  comm  = MPI_COMM_WORLD;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(comm, &rank);    
  MPI_Comm_size(comm, &size);
  MPI_Request request;   // for non-blicking send-receive later on
  MPI_Status status;

  int M = 201;  // M length intervals
  int N = 10000; // N time intervals
  int Jn = ((M+1)-2)/size + 2;   // remember M ->not #points, but #intervals
  double T = atof(argv[1]);  // get final time from input argument
  double U[2][M+1];  // stores the numerical values of function U; two rows to also store values of previous time step 
  double Ul[2][Jn];  // local array Ul that each of the processes works with
  double Usol[M+1];  // stores true solution 
  double dt = T/N;
  double dx = 1./M;
  double dtdx = dt/(dx*dx);
  double t1,t2;
  
  t1 =  MPI_Wtime();

  // initialize numerical array with given conditions
  U[0][0]=0, U[0][M]=0, U[1][0]=0, U[1][M]=0;        // U(t,x=0) = U(t,x=1) = 0
  for(int m=1; m<M; ++m){
	  U[0][m] = sin(2*PI*m*dx) + 2*sin(5*PI*m*dx) + 3*sin(20*PI*m*dx);
  }

  for (int m=0; m<Jn; m++){
      Ul[0][m]=U[0][rank*(Jn-2)+m];
   }

  if (rank == 0){    // left boundary condition
     Ul[1][0] = 0;
  }

  if (rank == size-1){ // right boundary condition
     Ul[1][Jn-1] = 0;
  }

  // use numerical scheme to obtain the future values of U on the M+1 space points
  for(int i=1; i<=N; ++i){
       	  for (int m=1; m<Jn-1; ++m){
  		   Ul[1][m] = Ul[0][m] + dtdx *(Ul[0][m-1] - 2*Ul[0][m] + Ul[0][m+1]);	
	  }
 
         
	  if (rank !=0){    // each process apart from the 1st one sends its 2nd
	                    // element to the previous one
             MPI_Send(&Ul[1][1],1, MPI_DOUBLE, rank-1, 2, comm);
	  }

          if (rank != size-1){  // each process apart from the last one receives 
	                        // the second element sent by the next process
             MPI_Recv(&Ul[1][Jn-1],1, MPI_DOUBLE, rank+1, 2, comm, MPI_STATUS_IGNORE);
          }

	  if (rank !=size-1){    // each process apart from the last one sends
	                         // its previous to last element to the next one
             MPI_Send(&Ul[1][Jn-2],1, MPI_DOUBLE, rank+1, 2, comm);
	  }

          if (rank != 0){  // each process apart from the first one receives 
	                        // the second element sent by the previous process
                MPI_Recv(&Ul[1][0],1, MPI_DOUBLE, rank-1, 2, comm, MPI_STATUS_IGNORE);
          }
        
	  for(int m=0; m<=Jn-1; m++){
   		  Ul[0][m] = Ul[1][m];
          }

  }

  if (rank==0){
     for (int m=0; m<Jn; m++){
         U[1][m] = Ul[1][m]; 
     }
  }
  
  if (rank!=0){
     MPI_Send(&Ul[1][0],Jn, MPI_DOUBLE, 0, 2, comm);
  }
  
  if (rank==0){
     for (int r=1; r<=size-1; r++){
         MPI_Recv(&U[1][r*(Jn-2)],Jn, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
     }
  }

  if (rank==0){
    cout<< "\ndx="<<dx<<", dt="<<dt<<", dt/dx²="<< dtdx<<endl;
    // print out array entries of numerical solution next to true solution
    cout << "\nTrue and numerical values at M="<<M<<" space points at time T="<<T<<":"<<endl;
    cout << "\nTrue values           Numerical solutions\n"<<endl;
    for(int m=0; m<=M; ++m){
   	    Usol[m] = exp(-4*PI*PI*T)*sin(2*PI*m*dx) + 2*exp(-25*PI*PI*T)*sin(5*PI*m*dx) + 3*exp(-400*PI*PI*T)*sin(20*PI*M*dx);		
	    cout << Usol[m] << "            " << U[1][m] << endl;
	    // note that we did not really need to store the true solution in the array just to print out the values.
    }
  }
  
  t2 = MPI_Wtime();
 
  if (rank==0){
     cout << "time to run= " << t2-t1 << endl;
  }
  MPI_Finalize();

}
