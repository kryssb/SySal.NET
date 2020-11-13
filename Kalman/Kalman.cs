using System;

namespace NumericalTools
{

	public enum FilterDirection  : int {Forward=1, Backward=-1}


	public class FilterStep
	{

		//  measurement vector
		protected double[] m_MeasureVec;
		public double[] MeasureVec { get { return m_MeasureVec; } }
		internal void SetMeasureVec(double[] input_vector)
		{
			int r = m_MeasureVec.Length;
			int i;
			if(r!=input_vector.Length) 
				throw new Exception("Measure Vector Dimension does not fit");
			for(i=0; i<r; i++)
				m_MeasureVec[i]=input_vector[i];
		}

		//  (filtered) state vector
		protected double[] m_StateVec;
		public double[] StateVec { get { return m_StateVec; } }
		internal void SetStateVec(double[] input_vector)
		{
			int r = m_StateVec.Length;
			int i;
			if(r!=input_vector.Length) 
				throw new Exception("State Vector Dimension does not fit");
			for(i=0; i<r; i++)
					m_StateVec[i]=input_vector[i];
		}

		//  predicted state vector
		protected double[] m_PredStateVec;
		public double[] PredStateVec { get { return m_PredStateVec; } }
		internal void SetPredStateVec(double[] input_vector)
		{
			int r = m_PredStateVec.Length;
			int i;
			if(r!=input_vector.Length) 
				throw new Exception("Predicted State Vector Dimension does not fit");
			for(i=0; i<r; i++)
				m_PredStateVec[i]=input_vector[i];
		}

		// state vector
		protected int m_StateVectorLength;
		public int StateVectorLength { get { return m_StateVectorLength; } }

		protected int m_MeasurementsVectorLength;
		public int MeasurementsVectorLength { get { return m_MeasurementsVectorLength; } }

		//forward propagator matrix
		protected double[,] m_ForwardPropagator;
		public double[,] ForwardPropagator { get { return m_ForwardPropagator; } }

		//backward propagator matrix 
		protected double[,] m_BackwardPropagator;
		public double[,] BackwardPropagator { get { return m_BackwardPropagator; } }

		//covariance of measurement error
		protected double[,] m_MeasCovariance;
		public double[,] MeasCovariance { get { return m_MeasCovariance; } }

		// covariance of process noise
		protected double[,] m_NoiseCovariance;
		public double[,] NoiseCovariance { get { return m_NoiseCovariance; } }

		protected double[,] m_StateCovariance;
		public double[,] StateCovariance { get { return m_StateCovariance; } }

		// predicted covariance of state
		protected double[,] m_PredStateCovariance;
		public double[,] PredStateCovariance { get { return m_PredStateCovariance; } }
		internal void SetPredictedStateCovariance(double[,] input_matrix)
		{
			int r = m_PredStateCovariance.GetLength(0);
			int c = m_PredStateCovariance.GetLength(1);
			int i,j;
			if(r!=input_matrix.GetLength(0) || c != input_matrix.GetLength(1)) 
				throw new Exception("Predicted State Covariance Matrix Dimensions do not fit");
			for(i=0; i<r; i++)
				for(j=0; j<c; j++)
					m_PredStateCovariance[i,j]=input_matrix[i,j];
		}

		// filtered covariance of state
		protected double[,] m_FiltStateCovariance;
		public double[,] FiltStateCovariance { get { return m_FiltStateCovariance; } }
		internal void SetFiltStateCovariance(double[,] input_matrix)
		{
			int r = m_FiltStateCovariance.GetLength(0);
			int c = m_FiltStateCovariance.GetLength(1);
			int i,j;
			if(r!=input_matrix.GetLength(0) || c != input_matrix.GetLength(1)) 
				throw new Exception("Filtered State Covariance Matrix Dimensions do not fit");
			for(i=0; i<r; i++)
				for(j=0; j<c; j++)
					m_FiltStateCovariance[i,j]=input_matrix[i,j];
		}

		// converts state vector to measurement vector
		protected double[,] m_StateMeasure;
		public double[,] StateMeasure { get { return m_StateMeasure; } }

		// kalman gain matrix
		protected double[,] m_KalmanGain;
		public double[,] KalmanGain { get { return m_KalmanGain; } }
		internal void SetKalmanGain(double[,] input_matrix)
		{
			int r = m_KalmanGain.GetLength(0);
			int c = m_KalmanGain.GetLength(1);
			int i,j;
			if(r!=input_matrix.GetLength(0) || c != input_matrix.GetLength(1)) 
				throw new Exception("Kalman Gain Matrix Dimensions do not fit");
			for(i=0; i<r; i++)
				for(j=0; j<c; j++)
					m_KalmanGain[i,j]=input_matrix[i,j];
		}
		
		// residual vector
		protected double[] m_ResidualVec;
		public double[] ResidualVec { get { return m_ResidualVec; } }
		internal void SetResidualVec(double[] input_vector)
		{
			int r = m_ResidualVec.Length;
			int i;
			if(r!=input_vector.Length) 
				throw new Exception("Residual Vector Dimension does not fit");
			for(i=0; i<r; i++)
				m_ResidualVec[i]=input_vector[i];
		}

		// residual matrix - identity
		protected double[,] m_ResidualMat;
		public double[,] ResidualMat { get { return m_ResidualMat; } }
		internal void SetResidualMat(double[,] input_matrix)
		{
			int r = m_ResidualMat.GetLength(0);
			int c = m_ResidualMat.GetLength(1);
			int i,j;
			if(r!=input_matrix.GetLength(0) || c != input_matrix.GetLength(1)) 
				throw new Exception("Residual Matrix Dimensions do not fit");
			for(i=0; i<r; i++)
				for(j=0; j<c; j++)
					m_ResidualMat[i,j]=input_matrix[i,j];
		}

		// chi**2
		protected double m_ChiSquared;
		public double ChiSquared { get { return m_ChiSquared; } }
		internal void SetChiSquared(double chisq) {m_ChiSquared=chisq;}

		//Index
		protected int m_Index;
		public int Index { get { return m_Index; } }

		public FilterStep(int index, double[] MeasVec, double[] StatVec)
		{

			m_Index = index;
			m_StateVectorLength = StatVec.Length;
			m_MeasurementsVectorLength = MeasVec.Length;
			int i;

			m_MeasureVec = new double[m_MeasurementsVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) m_MeasureVec[i]=MeasVec[i];
			m_StateVec = new double[m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) m_StateVec[i]=StatVec[i];
			m_PredStateVec = new double[m_StateVectorLength];

			m_ForwardPropagator = Matrices.Identity(m_StateVectorLength);
			m_BackwardPropagator = Matrices.Identity(m_StateVectorLength);

			m_KalmanGain = new double[m_StateVectorLength, m_MeasurementsVectorLength];

			m_StateMeasure = new double[m_MeasurementsVectorLength,m_StateVectorLength];

			m_PredStateCovariance = Matrices.Identity(m_StateVectorLength);
			m_FiltStateCovariance = Matrices.Identity(m_StateVectorLength);

			m_NoiseCovariance = new double[m_StateVectorLength,m_StateVectorLength];
			m_MeasCovariance = Matrices.Identity(m_MeasurementsVectorLength);

			m_ResidualVec = new double[m_MeasurementsVectorLength];
			m_ResidualMat = Matrices.Identity(m_MeasurementsVectorLength);
			m_ChiSquared = 0;

		}

		public FilterStep(int index, double[] MeasVec, double[] StatVec, double[,] ForwProp, double[,] BackProp, double[,] MeasCov, double[,] NoisCov,
			double[,] StatMeas, double[,] FiltCov)
		{
			int i,j;

			m_Index = index;
			m_StateVectorLength = StatVec.Length;
			m_MeasurementsVectorLength = MeasVec.Length;

			m_MeasureVec = new double[m_MeasurementsVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) m_MeasureVec[i]=MeasVec[i];
			m_StateVec = new double[m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) m_StateVec[i]=StatVec[i];
			m_PredStateVec = new double[m_StateVectorLength];

			//Controllo
			if(ForwProp.GetLength(0) != m_StateVectorLength || ForwProp.GetLength(1) != m_StateVectorLength) throw new Exception("Step " + index + ": Forward Propagator dimensions do not fit");
			m_ForwardPropagator = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_ForwardPropagator[i,j]=ForwProp[i,j];
			if(BackProp.GetLength(0) != m_StateVectorLength || BackProp.GetLength(1) != m_StateVectorLength) throw new Exception("Step " + index + ": Backward Propagator dimensions do not fit");
			m_BackwardPropagator = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_BackwardPropagator[i,j]=BackProp[i,j];

			m_KalmanGain = new double[m_StateVectorLength, m_MeasurementsVectorLength];

			if(StatMeas.GetLength(0) != m_MeasurementsVectorLength || StatMeas.GetLength(1) != m_StateVectorLength) throw new Exception("Step " + index + ": State-Measure Matrix dimensions do not fit");
			m_StateMeasure = new double[m_MeasurementsVectorLength,m_StateVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) for(j=0; j<StateVectorLength; j++) m_StateMeasure[i,j]=StatMeas[i,j];

			m_PredStateCovariance = Matrices.Identity(m_StateVectorLength);
			if(FiltCov.GetLength(0) != m_StateVectorLength || FiltCov.GetLength(1) != m_StateVectorLength) throw new Exception("Step " + index + ": Filtered Covariance Matrix dimensions do not fit");
			m_FiltStateCovariance = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_FiltStateCovariance[i,j]=FiltCov[i,j];

			if(NoisCov.GetLength(0) != m_StateVectorLength || NoisCov.GetLength(1) != m_StateVectorLength) throw new Exception("Step " + index + ": Noise Covariance Matrix dimensions do not fit");
			m_NoiseCovariance = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_NoiseCovariance[i,j]=NoisCov[i,j];
			if(MeasCov.GetLength(0) != m_MeasurementsVectorLength || MeasCov.GetLength(1) != m_MeasurementsVectorLength) throw new Exception("Step " + index + ": Measure Covariance Matrix dimensions do not fit");
			m_MeasCovariance = new double[m_MeasurementsVectorLength,m_MeasurementsVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) for(j=0; j<m_MeasurementsVectorLength; j++) m_MeasCovariance[i,j]=MeasCov[i,j];

			m_ResidualVec = new double[m_MeasurementsVectorLength];
			m_ResidualMat = Matrices.Identity(m_MeasurementsVectorLength);
			m_ChiSquared = 0;

		}

		public FilterStep(int index, FilterStep fs)
		{
			int i,j;

			m_Index = index;
			m_StateVectorLength = fs.StateVec.Length;
			m_MeasurementsVectorLength = fs.MeasCovariance.GetLength(0);

			m_MeasureVec = new double[m_MeasurementsVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) m_MeasureVec[i]=fs.MeasureVec[i];
			m_StateVec = new double[m_StateVectorLength];
			m_PredStateVec = new double[m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) m_StateVec[i]=fs.StateVec[i];

			m_ForwardPropagator = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_ForwardPropagator[i,j]= fs.ForwardPropagator[i,j];
			m_BackwardPropagator = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_BackwardPropagator[i,j]=fs.BackwardPropagator[i,j];

			m_KalmanGain = new double[m_StateVectorLength, m_MeasurementsVectorLength];

			m_StateMeasure = new double[m_MeasurementsVectorLength,m_StateVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) for(j=0; j<StateVectorLength; j++) m_StateMeasure[i,j]=fs.StateMeasure[i,j];

			m_PredStateCovariance = Matrices.Identity(m_StateVectorLength);
			m_FiltStateCovariance = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_FiltStateCovariance[i,j]=fs.FiltStateCovariance[i,j];

			m_NoiseCovariance = new double[m_StateVectorLength,m_StateVectorLength];
			for(i=0; i<m_StateVectorLength; i++) for(j=0; j<m_StateVectorLength; j++) m_NoiseCovariance[i,j]=fs.NoiseCovariance[i,j];
			m_MeasCovariance = new double[m_MeasurementsVectorLength,m_MeasurementsVectorLength];
			for(i=0; i<m_MeasurementsVectorLength; i++) for(j=0; j<m_MeasurementsVectorLength; j++) m_MeasCovariance[i,j]=fs.MeasCovariance[i,j];

			m_ResidualVec = new double[m_MeasurementsVectorLength];
			m_ResidualMat = Matrices.Identity(m_MeasurementsVectorLength);
			m_ChiSquared = 0;

		}
	}

	/// <summary>
	/// A list of Kalman Steps. 
	/// </summary>
	public class FilterStepList
	{
		/// <summary>
		/// Member data holding the list of steps. Can be accessed by derived classes.
		/// </summary>
		protected FilterStep[] FilterSteps = new FilterStep[0];

		/// <summary>
		/// Accesses the index-th step.
		/// </summary>
		public FilterStep this[int index]
		{
			get { return FilterSteps[index];  }
		}

		/// <summary>
		/// Number of steps in the list.
		/// </summary>
		public int Length
		{
			get { return FilterSteps.Length; }
		}

	}

	/// <summary>
	/// Summary description for Kalman.
	/// </summary>
	public class KalmanFilter : FilterStepList
	{
		public KalmanFilter()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		public int StepsNumber { get { return FilterSteps.Length; } }
		public int StateVectorLength { get { return FilterSteps[0].StateVectorLength; } }
		public int MeasurementVectorLength { get { return FilterSteps[0].MeasurementsVectorLength; } }
		public FilterDirection LastFilter = FilterDirection.Backward;

		public void AddConstantMatricesSteps(double[,] MeasuresInRows, double[,] ForwProp, double[,] BackProp, double[,] MeasCov, double[,] NoisCov,
			double[,] StatMeas, double[,] FiltCov)
		{
			int i,j;
			int n = MeasuresInRows.GetLength(1);
			double[] tmpvec = new double[n];

			int m = MeasuresInRows.GetLength(0);
			FilterSteps = new FilterStep[m];

			//bisogna manipolare l'indice
			for (i=0; i < m; i++)
			{
				for(j=0; j<n; j++) tmpvec[j]= MeasuresInRows[i,j];
				FilterSteps[i] = new FilterStep(i, tmpvec, tmpvec, ForwProp, BackProp, MeasCov, NoisCov, StatMeas, FiltCov);
			}
		}

		public void AddConstantMatricesSteps(double[,] MeasuresInRows, double[,] StatesInRows, double[,] ForwProp, double[,] BackProp, double[,] MeasCov, double[,] NoisCov,
			double[,] StatMeas, double[,] FiltCov)
		{
			int i,j;
			int nm = MeasuresInRows.GetLength(1);
			double[] tmpvec = new double[nm];

			int ns = StatesInRows.GetLength(1);
			double[] tmpvec2 = new double[ns];

			int m = MeasuresInRows.GetLength(0);
			FilterSteps = new FilterStep[m];

			//bisogna manipolare l'indice
			for (i=0; i < m; i++)
			{
				for(j=0; j<nm; j++) tmpvec[j]= MeasuresInRows[i,j];
				for(j=0; j<ns; j++) tmpvec2[j]= StatesInRows[i,j];
				FilterSteps[i] = new FilterStep(i, tmpvec, tmpvec2, 
					ForwProp, BackProp, MeasCov, NoisCov, StatMeas, FiltCov);
			}
		}

		public void AddStep(FilterStep s)
		{
			int i, k=0;
			FilterStep[] tmp;


			if(FilterSteps.Length != 0)
			{
				//Controlli sulla coerenza delle matrici degli steps
				if(s.MeasurementsVectorLength!= FilterSteps[0].MeasurementsVectorLength || 
					s.StateVectorLength!= FilterSteps[0].StateVectorLength ) 
					throw new Exception("Incoherent Step Added: \n\r State Vector Length or Measurement Vector Length are different from previous steps!");
				//Controlli sull'indice del nuovo step
				for (i=0; i < FilterSteps.Length; i++) 
					if (FilterSteps[i].Index==s.Index) 
						throw new Exception("Duplicate Index: \n\r Filter Step added has the same index of Filter Step #" + i);

				tmp = FilterSteps;
				FilterSteps = new FilterStep[FilterSteps.Length +1];
				for (i=0; i < tmp.Length; i++) if (tmp[i].Index<s.Index) k++;

				FilterSteps[k] = s;

				for (i = 0; i < k; i++) FilterSteps[i] = tmp[i];
				for (; i < FilterSteps.Length-1; i++) FilterSteps[i + 1] = tmp[i];
			}
			else FilterSteps = new FilterStep[1] {s};

		}

		public void RemoveStep(int StepIndex)
		{
			int i;
			for (i = 0; i < FilterSteps.Length && FilterSteps[i].Index != StepIndex; i++);
			if (i == FilterSteps.Length) throw new Exception("No step to remove");

			FilterStep [] tmp = FilterSteps;
			FilterSteps = new FilterStep[tmp.Length - 1];
			for (; i < FilterSteps.Length; i++)
				FilterSteps[i] = tmp[i + 1];

		}


		public void NextFilter()
		{
			if(LastFilter == FilterDirection.Backward) FilterForward();
			else FilterBackward();
		}
		
		public void FilterForward()
		{
			int j,i,n = FilterSteps.Length;
			int nstvec = FilterSteps[0].StateVectorLength;
			int nmsvec = FilterSteps[0].MeasurementsVectorLength;
			if (n<=0 || nstvec<=0 || nmsvec<=0) return;
			for (i=1; i</*=*/n/*+1*/; i++) 
			{
				// predict the covariance of estimate
				double[,] Cmm = Matrices.Product(FilterSteps[i-1].ForwardPropagator,FilterSteps[i-1].FiltStateCovariance);
				double[,] Amm = Matrices.Transpose(FilterSteps[i-1].ForwardPropagator);
				Cmm = Matrices.Product(Cmm,Amm);
				Cmm = Matrices.Sum(Cmm,FilterSteps[i-1].NoiseCovariance);
				FilterSteps[i].SetPredictedStateCovariance(Cmm);

				// calculate kalman gain matrix
				double[,] Cnm = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].PredStateCovariance);
				double[,] Bmn = Matrices.Transpose(FilterSteps[i].StateMeasure);
				double[,] Cnn = Matrices.Product(Cnm,Bmn);
				Cnn = Matrices.Sum(Cnn,FilterSteps[i].MeasCovariance);
				double[,] Cnninv = Matrices.InvertMatrix(Cnn);
				double[,] Cmn = Matrices.Product(FilterSteps[i].PredStateCovariance,Bmn);
				FilterSteps[i].SetKalmanGain(Matrices.Product(Cmn,Cnninv));

				// predict and filter state vector
				if (i<=n) 
				{
					// if there is a measurement
					double[] xm = Matrices.Product(FilterSteps[i-1].ForwardPropagator,FilterSteps[i-1].StateVec);
					FilterSteps[i].SetPredStateVec(xm);
					double[] ym = new double[xm.Length];
					for(j=0; j< xm.Length; j++) ym[j]=xm[j];
					double[] xn = (Matrices.Product(FilterSteps[i].StateMeasure,xm));
					xn = Matrices.Subtract(FilterSteps[i].MeasureVec,xn);
					xm = Matrices.Product(FilterSteps[i].KalmanGain,xn);
					xm = Matrices.Sum(xm,ym);
					FilterSteps[i].SetStateVec(xm);
				} 
				else 
				{
					// no measurement (i.e. final iteration)
					FilterSteps[i].SetStateVec(Matrices.Product(FilterSteps[i-1].ForwardPropagator,FilterSteps[i-1].StateVec));
				}

				// filter covariance matrix
				Cmm = Matrices.Product(FilterSteps[i].KalmanGain,FilterSteps[i].StateMeasure);
				double[,] Bmm = Matrices.Identity(nstvec);
				Bmm = Matrices.Subtract(Bmm,Cmm);
				Bmm = Matrices.Product(Bmm,FilterSteps[i].PredStateCovariance);
				FilterSteps[i].SetFiltStateCovariance(Bmm);
				
				//Quality checks on filtering progress
				if (i<=n) 
				{
					// calculate filtered residual
					double[] zn = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].StateVec);
					zn = Matrices.Subtract(FilterSteps[i].MeasureVec,zn);
					FilterSteps[i].SetResidualVec(zn);
					
					// calculate residual covariance matrix
					Cnn = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].KalmanGain);
					double[,] Bnn = Matrices.Identity(nmsvec);
					Bnn = Matrices.Subtract(Bnn,Cnn);
					Bnn = Matrices.Product(Bnn,FilterSteps[i].MeasCovariance);
					FilterSteps[i].SetResidualMat(Bnn);
					
					// calculate chi-squared
					double[,] Dnn = Matrices.InvertMatrix(FilterSteps[i].ResidualMat);
					zn = Matrices.Product(Dnn,FilterSteps[i].ResidualVec);
					FilterSteps[i].SetChiSquared(Matrices.ScalarProduct(FilterSteps[i].ResidualVec,zn));
				}
			}
			LastFilter = FilterDirection.Forward;
	
		}
		
		public void FilterBackward()
		{
			int j,i,n = FilterSteps.Length;
			int nstvec = FilterSteps[0].StateVectorLength;
			int nmsvec = FilterSteps[0].MeasurementsVectorLength;
			if (n<=0 || nstvec<=0 || nmsvec<=0) return;
			//for (i=n; i>=0; i--) 
			for (i=n-2; i>=0; i--) 
			{
				// predict the covariance of estimate
				double[,] Cmm = Matrices.Product(FilterSteps[i+1].BackwardPropagator,FilterSteps[i+1].FiltStateCovariance);
				double[,] Amm = Matrices.Transpose(FilterSteps[i+1].BackwardPropagator);
				Cmm = Matrices.Product(Cmm,Amm);
				Cmm = Matrices.Sum(Cmm,FilterSteps[i+1].NoiseCovariance);
				FilterSteps[i].SetPredictedStateCovariance(Cmm);

				// calculate kalman gain matrix
				double[,] Cnm = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].PredStateCovariance);
				double[,] Bmn = Matrices.Transpose(FilterSteps[i].StateMeasure);
				double[,] Cnn = Matrices.Product(Cnm,Bmn);
				Cnn = Matrices.Sum(Cnn,FilterSteps[i].MeasCovariance);
				double[,] Cnninv = Matrices.InvertMatrix(Cnn);
				double[,] Cmn = Matrices.Product(FilterSteps[i].PredStateCovariance,Bmn);
				FilterSteps[i].SetKalmanGain(Matrices.Product(Cmn,Cnninv));

				// predict and filter state vector
				if (i>0) 
				{
					// if there is a measurement
					double[] xm = Matrices.Product(FilterSteps[i+1].BackwardPropagator,FilterSteps[i+1].StateVec);
					FilterSteps[i].SetPredStateVec(xm);
					double[] ym = new double[xm.Length];
					for(j=0; j< xm.Length; j++) ym[j]=xm[j];
					double[] xn = Matrices.Product(FilterSteps[i].StateMeasure,xm);
					xn = Matrices.Subtract(FilterSteps[i].MeasureVec,xn);
					xm = Matrices.Product(FilterSteps[i].KalmanGain,xn);
					xm = Matrices.Sum(xm,ym);
					FilterSteps[i].SetStateVec(xm);
				} 
				else 
				{
					// no measurement (i.e. final iteration)
					FilterSteps[i].SetStateVec(Matrices.Product(FilterSteps[i+1].BackwardPropagator,FilterSteps[i+1].StateVec));
				}

				// filter covariance matrix
				Cmm = Matrices.Product(FilterSteps[i].KalmanGain,FilterSteps[i].StateMeasure);
				double[,] Bmm = Matrices.Identity(nstvec);
				Bmm = Matrices.Subtract(Bmm,Cmm);
				Bmm = Matrices.Product(Bmm,FilterSteps[i].PredStateCovariance);
				FilterSteps[i].SetFiltStateCovariance(Bmm);

				//Quality checks on filtering progress
				if (i<=n) 
				{
					// calculate filtered residual
					double[] zn = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].StateVec);
					zn = Matrices.Subtract(FilterSteps[i].MeasureVec,zn);
					FilterSteps[i].SetResidualVec(zn);

					// calculate residual covariance matrix
					Cnn = Matrices.Product(FilterSteps[i].StateMeasure,FilterSteps[i].KalmanGain);
					double[,] Bnn = Matrices.Identity(nmsvec);
					Bnn = Matrices.Subtract(Bnn,Cnn);
					Bnn = Matrices.Product(Bnn,FilterSteps[i].MeasCovariance);
					FilterSteps[i].SetResidualMat(Bnn);

					// calculate chi-squared
					double[,] Dnn = Matrices.InvertMatrix(FilterSteps[i].ResidualMat);
					zn = Matrices.Product(Dnn,FilterSteps[i].ResidualVec);
					FilterSteps[i].SetChiSquared(Matrices.ScalarProduct(FilterSteps[i].ResidualVec,zn));
				}
			}
			LastFilter = FilterDirection.Forward;
		
		}

	}
}
