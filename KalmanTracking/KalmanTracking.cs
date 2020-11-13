using System;
using NumericalTools;
using SySal.Tracking;

namespace SySal.TotalScan
{
	public enum ProjectionDirection  : int {DownStream=1, UpStream=-1}
	/// <summary>
	/// Summary description for KalmanTracking Class.
	/// </summary>
	public abstract class KalmanTracking
	{
		protected KalmanFilter kf;
		protected KalmanTracking()
		{
			//
			// TODO: Add constructor logic here
			//
		}


		public abstract void SetVolumeTrack(Track t);

		public abstract MIPEmulsionTrackInfo ProjectVolumeTrackByGap(ProjectionDirection direct, double projection_gap);

		public abstract MIPEmulsionTrackInfo ProjectVolumeTrackByGap(Track t, ProjectionDirection direct, double projection_gap);

		public abstract MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(ProjectionDirection direct, double projection_Z);

		public abstract MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(Track t, ProjectionDirection direct, double projection_Z);
	}
#if false
	public class SimpleKalmanTracking: KalmanTracking
	{

		private double m_ProjectionZ;
		private Track m_Track = null;
		private double m_ErrorX;
		private double m_ErrorY;
		private double m_ErrorSx;
		private double m_ErrorSy;
		private double m_NoiseX;
		private double m_NoiseY;
		private double m_NoiseSx;
		private double m_NoiseSy;
		private double[,] m_fp;
		private double[,] m_bp;
		private double[,] m_mc;
		private double[,] m_nc;
		private double[,] m_st;
		private double[,] m_fc;

		private void SetKalmanFilter()
		{
			int j,n;
			n = m_Track.Length;
			int ng;
			kf = new KalmanFilter();

			double[] mrows = new double[4];
			for (j=0; j<n; j++)
			{

				mrows[0] = m_Track[j].Info.Intercept.X;
				mrows[1] = m_Track[j].Info.Slope.X;
				mrows[2] = m_Track[j].Info.Intercept.Y;
				mrows[3] = m_Track[j].Info.Slope.Y;
				if(j==n-1)
				{
					ng = 1;
				}
				else
				{
					ng = Math.Abs(m_Track[j+1].LayerOwner.SheetId - m_Track[j].LayerOwner.SheetId);
				}
				m_fp = new double[4,4] {{1,-ng*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,-ng*m_ProjectionZ},{0,0,0,1}};
				m_bp = new double[4,4] {{1,ng*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,ng*m_ProjectionZ},{0,0,0,1}};
				FilterStep fs = new FilterStep(j, mrows, mrows, m_fp, m_bp, m_mc, m_nc, m_st, m_fc);
				kf.AddStep(fs);
			}
		
		}

		public SimpleKalmanTracking(double ErrorX, double ErrorY, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy, double ProjectionZ)
		{

			//Assignment to private variables
			m_ProjectionZ = ProjectionZ;
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[4,4];
			m_bp = new double[4,4];
			m_mc = new double[4,4] {{ErrorX,0,0,0},{0,ErrorSx,0,0},{0,0,ErrorY,0},{0,0,0,ErrorSy}};
			m_nc = new double[4,4] {{NoiseX,0,0,0},{0,NoiseSx,0,0},{0,0,NoiseY,0},{0,0,0,NoiseSy}};
			m_st = new double[4,4] {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
			m_fc = new double[4,4] {{0.01,0,0,0},{0,0.01,0,0},{0,0,0.01,0},{0,0,0,0.01}};

		}


		public SimpleKalmanTracking(Track t, double ErrorX, double ErrorY, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy, double ProjectionZ)
		{
			

			//Assignment to private variables
			m_Track = t;
			m_ProjectionZ = ProjectionZ;
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[4,4];
			m_bp = new double[4,4];
			m_mc = new double[4,4] {{ErrorX,0,0,0},{0,ErrorSx,0,0},{0,0,ErrorY,0},{0,0,0,ErrorSy}};
			m_nc = new double[4,4] {{NoiseX,0,0,0},{0,NoiseSx,0,0},{0,0,NoiseY,0},{0,0,0,NoiseSy}};
			m_st = new double[4,4] {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
			m_fc = new double[4,4] {{0.01,0,0,0},{0,0.01,0,0},{0,0,0.01,0},{0,0,0,0.01}};

			SetKalmanFilter();
		}

		public override void SetVolumeTrack(Track t)
		{
			m_Track = t;
			SetKalmanFilter();
		
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrack(ProjectionDirection direct, int layernumber)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			if(m_Track==null) throw new Exception("Track not set!");
			if (layernumber<1) throw new Exception("Layer Number must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[4,4] {{1,layernumber*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,layernumber*m_ProjectionZ},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = layernumber*m_ProjectionZ + m_Track[0].Info.BottomZ;
				mipt.TopZ = layernumber*m_ProjectionZ + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = layernumber*m_ProjectionZ + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[4,4] {{1,-layernumber*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,-layernumber*m_ProjectionZ},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -layernumber*m_ProjectionZ + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -layernumber*m_ProjectionZ + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -layernumber*m_ProjectionZ + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrack(Track t, ProjectionDirection direct, int layernumber)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			m_Track = t;
			SetKalmanFilter();

			if (layernumber<1) throw new Exception("Layer Number must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[4,4] {{1,layernumber*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,layernumber*m_ProjectionZ},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = layernumber*m_ProjectionZ + m_Track[0].Info.BottomZ;
				mipt.TopZ = layernumber*m_ProjectionZ + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = layernumber*m_ProjectionZ + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
				mipt.Count = m_Track[0].Info.Count;
				mipt.Field = m_Track[0].Info.Field;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[4,4] {{1,-layernumber*m_ProjectionZ,0,0},{0,1,0,0},{0,0,1,-layernumber*m_ProjectionZ},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -layernumber*m_ProjectionZ + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -layernumber*m_ProjectionZ + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -layernumber*m_ProjectionZ + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
				mipt.Count = m_Track[n-1].Info.Count;
				mipt.Field = m_Track[n-1].Info.Field;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}

	}
#endif
	public class SimpleKalmanTracking: KalmanTracking
	{

		private double m_ProjectionZ;
		private Track m_Track = null;
		private double m_ErrorX;
		private double m_ErrorY;
		private double m_ErrorSx;
		private double m_ErrorSy;
		private double m_NoiseX;
		private double m_NoiseY;
		private double m_NoiseSx;
		private double m_NoiseSy;
		private double[,] m_fp;
		private double[,] m_bp;
		private double[,] m_mc;
		private double[,] m_nc;
		private double[,] m_st;
		private double[,] m_fc;

		private void SetKalmanFilter()
		{
			int j,n;
			n = m_Track.Length;
			//int ng;
			double projz;
			kf = new KalmanFilter();

			double[] mrows = new double[4];
			for (j=0; j<n; j++)
			{

				mrows[0] = m_Track[j].Info.Intercept.X;
				mrows[1] = m_Track[j].Info.Slope.X;
				mrows[2] = m_Track[j].Info.Intercept.Y;
				mrows[3] = m_Track[j].Info.Slope.Y;
				/*if(j==n-1)
				{
					ng = 1;
				}
				else
				{
					ng = Math.Abs(m_Track[j+1].LayerOwner.SheetId - m_Track[j].LayerOwner.SheetId);
				}
				*/
				if(j==n-1)
				{
					if(j>0)
						projz = Math.Abs(m_Track[j].Info.Intercept.Z - m_Track[j-1].Info.Intercept.Z);
					else
						projz = m_ProjectionZ;
				}
				else
				{
					projz = Math.Abs(m_Track[j+1].Info.Intercept.Z - m_Track[j].Info.Intercept.Z);
				
				}
				m_fp = new double[4,4] {{1,-projz,0,0},{0,1,0,0},{0,0,1,-projz},{0,0,0,1}};
				m_bp = new double[4,4] {{1,projz,0,0},{0,1,0,0},{0,0,1,projz},{0,0,0,1}};
				FilterStep fs = new FilterStep(j, mrows, mrows, m_fp, m_bp, m_mc, m_nc, m_st, m_fc);
				kf.AddStep(fs);
			}
		
		}

		public SimpleKalmanTracking(double ErrorX, double ErrorY, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy, double OneMicroTrackProjectionZ)
		{

			//Assignment to private variables
			m_ProjectionZ = OneMicroTrackProjectionZ;
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[4,4];
			m_bp = new double[4,4];
			m_mc = new double[4,4] {{ErrorX,0,0,0},{0,ErrorSx,0,0},{0,0,ErrorY,0},{0,0,0,ErrorSy}};
			m_nc = new double[4,4] {{NoiseX,0,0,0},{0,NoiseSx,0,0},{0,0,NoiseY,0},{0,0,0,NoiseSy}};
			m_st = new double[4,4] {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
			m_fc = new double[4,4] {{0.01,0,0,0},{0,0.01,0,0},{0,0,0.01,0},{0,0,0,0.01}};

		}


		public SimpleKalmanTracking(Track t, double ErrorX, double ErrorY, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy, double OneMicroTrackProjectionZ)
		{
			

			//Assignment to private variables
			m_Track = t;
			m_ProjectionZ = OneMicroTrackProjectionZ;
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[4,4];
			m_bp = new double[4,4];
			m_mc = new double[4,4] {{ErrorX,0,0,0},{0,ErrorSx,0,0},{0,0,ErrorY,0},{0,0,0,ErrorSy}};
			m_nc = new double[4,4] {{NoiseX,0,0,0},{0,NoiseSx,0,0},{0,0,NoiseY,0},{0,0,0,NoiseSy}};
			m_st = new double[4,4] {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
			m_fc = new double[4,4] {{0.01,0,0,0},{0,0.01,0,0},{0,0,0.01,0},{0,0,0,0.01}};

			SetKalmanFilter();
		}

		public override void SetVolumeTrack(Track t)
		{
			m_Track = t;
			SetKalmanFilter();
		
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackByGap(ProjectionDirection direct, double projection_gap)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			if(m_Track==null) throw new Exception("Track not set!");
			if (projection_gap<0) throw new Exception("Projection Gap must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[4,4] {{1,projection_gap/*layernumber*m_ProjectionZ*/,0,0},{0,1,0,0},{0,0,1,projection_gap/*layernumber*m_ProjectionZ*/},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[4,4] {{1,-projection_gap/*layernumber*m_ProjectionZ*/,0,0},{0,1,0,0},{0,0,1,-projection_gap/*layernumber*m_ProjectionZ*/},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackByGap(Track t, ProjectionDirection direct, double projection_gap)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			m_Track = t;
			SetKalmanFilter();

			if (projection_gap<0) throw new Exception("Projection Gap must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[4,4] {{1,projection_gap/*layernumber*m_ProjectionZ*/,0,0},{0,1,0,0},{0,0,1,projection_gap/*layernumber*m_ProjectionZ*/},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
				mipt.Count = m_Track[0].Info.Count;
				mipt.Field = m_Track[0].Info.Field;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[4,4] {{1,-projection_gap/*layernumber*m_ProjectionZ*/,0,0},{0,1,0,0},{0,0,1,-projection_gap/*layernumber*m_ProjectionZ*/},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -projection_gap/*layernumber*m_ProjectionZ*/ + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
				mipt.Count = m_Track[n-1].Info.Count;
				mipt.Field = m_Track[n-1].Info.Field;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(ProjectionDirection direct, double projection_Z)
		{
			double[] tvec;
			double[,] prop;
			double projection_gap;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			if(m_Track==null) throw new Exception("Track not set!");
			//if (layernumber<1) throw new Exception("Layer Number must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				projection_gap = projection_Z - m_Track[0].Info.Intercept.Z;
				prop = new double[4,4] {{1,projection_gap,0,0},{0,1,0,0},{0,0,1,projection_gap},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = projection_gap + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				projection_gap =  m_Track[n-1].Info.Intercept.Z - projection_Z;
				prop = new double[4,4] {{1,-projection_gap,0,0},{0,1,0,0},{0,0,1,-projection_gap},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -projection_gap + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(Track t, ProjectionDirection direct, double projection_Z)
		{
			double[] tvec;
			double[,] prop;
			double projection_gap;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			m_Track = t;
			SetKalmanFilter();

			//if (layernumber<1) throw new Exception("Layer Number must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				projection_gap = projection_Z - m_Track[0].Info.Intercept.Z;
				prop = new double[4,4] {{1,projection_gap,0,0},{0,1,0,0},{0,0,1,projection_gap},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.Intercept.Z = projection_gap + m_Track[0].Info.Intercept.Z;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
				mipt.Count = m_Track[0].Info.Count;
				mipt.Field = m_Track[0].Info.Field;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				projection_gap =  m_Track[n-1].Info.Intercept.Z - projection_Z;
				prop = new double[4,4] {{1,-projection_gap,0,0},{0,1,0,0},{0,0,1,-projection_gap},{0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.Intercept.Z = -projection_gap + m_Track[n-1].Info.Intercept.Z;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
				mipt.Count = m_Track[n-1].Info.Count;
				mipt.Field = m_Track[n-1].Info.Field;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Slope.Z = 1;

			return mipt;
		}
	}

	public class KalmanTracking3D: KalmanTracking
	{

		private Track m_Track = null;
		private double m_ErrorX;
		private double m_ErrorY;
		private double m_ErrorZ;
		private double m_ErrorSx;
		private double m_ErrorSy;
		private double m_NoiseX;
		private double m_NoiseY;
		private double m_NoiseSx;
		private double m_NoiseSy;
		private double[,] m_fp;
		private double[,] m_bp;
		private double[,] m_mc;
		private double[,] m_nc;
		private double[,] m_st;
		private double[,] m_fc;

		private void SetKalmanFilter()
		{
			int j,n;
			n = m_Track.Length;
			double projz;
			kf = new KalmanFilter();

			double[] srows = new double[6];
			double[] mrows = new double[5];
			for (j=0; j<n; j++)
			{

				srows[0] = m_Track[j].Info.Intercept.X;
				srows[1] = m_Track[j].Info.Slope.X;
				srows[2] = m_Track[j].Info.Intercept.Y;
				srows[3] = m_Track[j].Info.Slope.Y;
				srows[4] = m_Track[j].Info.Intercept.Z;
				srows[5] = 1;

				mrows[0] = srows[0];
				mrows[1] = srows[1];
				mrows[2] = srows[2];
				mrows[3] = srows[3];
				mrows[4] = srows[4];

				if(j==n-1)
				{
					projz = Math.Abs(m_Track[j].Info.Intercept.Z - m_Track[j-1].Info.Intercept.Z);
				}
				else
				{
					projz = Math.Abs(m_Track[j+1].Info.Intercept.Z - m_Track[j].Info.Intercept.Z);
				
				}
				m_fp = new double[6,6] {{1,-projz,0,0,0,0},{0,1,0,0,0,0},{0,0,1,-projz,0,0},{0,0,0,1,0,0},{0,0,0,0,1,-projz},{0,0,0,0,0,1}};
				m_bp = new double[6,6] {{1,projz,0,0,0,0},{0,1,0,0,0,0},{0,0,1,projz,0,0},{0,0,0,1,0,0},{0,0,0,0,1,projz},{0,0,0,0,0,1}};
				FilterStep fs = new FilterStep(j, mrows, srows, m_fp, m_bp, m_mc, m_nc, m_st, m_fc);
				kf.AddStep(fs);
			}
		
		}

		public KalmanTracking3D(double ErrorX, double ErrorY, double ErrorZ, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy)
		{

			//Assignment to private variables
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorZ = ErrorZ;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[6,6];
			m_bp = new double[6,6];
			m_mc = new double[5,5] {{ErrorX,0,0,0,0},{0,ErrorSx,0,0,0},{0,0,ErrorY,0,0},{0,0,0,ErrorSy,0},{0,0,0,0,ErrorZ}};
			m_nc = new double[6,6] {{NoiseX,0,0,0,0,0},{0,NoiseSx,0,0,0,0},{0,0,NoiseY,0,0,0},{0,0,0,NoiseSy,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}};
			m_st = new double[5,6] {{1,0,0,0,0,0},{0,1,0,0,0,0},{0,0,1,0,0,0},{0,0,0,1,0,0},{0,0,0,0,1,0}};
			m_fc = new double[6,6] {{0.01,0,0,0,0,0},{0,0.01,0,0,0,0},{0,0,0.01,0,0,0},{0,0,0,0.01,0,0},{0,0,0,0,0.01,0},{0,0,0,0,0,0.01}};

		}


		public KalmanTracking3D(Track t, double ErrorX, double ErrorY, double ErrorZ, double ErrorSx, double ErrorSy,
			double NoiseX, double NoiseY, double NoiseSx, double NoiseSy)
		{
			

			//Assignment to private variables
			m_Track = t;
			m_ErrorX = ErrorX;
			m_ErrorY = ErrorY;
			m_ErrorZ = ErrorZ;
			m_ErrorSx = ErrorSx;
			m_ErrorSy = ErrorSy;
			m_NoiseX = NoiseX;
			m_NoiseY = NoiseY;
			m_NoiseSx = NoiseSx;
			m_NoiseSy = NoiseSy;
			kf = new KalmanFilter();
			m_fp = new double[6,6];
			m_bp = new double[6,6];
			m_mc = new double[5,5] {{ErrorX,0,0,0,0},{0,ErrorSx,0,0,0},{0,0,ErrorY,0,0},{0,0,0,ErrorSy,0},{0,0,0,0,ErrorZ}};
			m_nc = new double[6,6] {{NoiseX,0,0,0,0,0},{0,NoiseSx,0,0,0,0},{0,0,NoiseY,0,0,0},{0,0,0,NoiseSy,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}};
			m_st = new double[5,6] {{1,0,0,0,0,0},{0,1,0,0,0,0},{0,0,1,0,0,0},{0,0,0,1,0,0},{0,0,0,0,1,0}};
			m_fc = new double[6,6] {{0.01,0,0,0,0,0},{0,0.01,0,0,0,0},{0,0,0.01,0,0,0},{0,0,0,0.01,0,0},{0,0,0,0,0.01,0},{0,0,0,0,0,0.01}};

			SetKalmanFilter();
		}

		public override void SetVolumeTrack(Track t)
		{
			m_Track = t;
			SetKalmanFilter();
		
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackByGap(ProjectionDirection direct, double projection_gap)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			if(m_Track==null) throw new Exception("Track not set!");
			if (projection_gap<0) throw new Exception("Projection Gap must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[6,6] {{1,projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[6,6] {{1,-projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,-projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,-projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Intercept.Z = tvec[4];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackByGap(Track t, ProjectionDirection direct, double projection_gap)
		{
			double[] tvec;
			double[,] prop;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			m_Track = t;
			SetKalmanFilter();

			if (projection_gap<0) throw new Exception("Projection Gap must be greater than 0");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				prop = new double[6,6] {{1,projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
				mipt.Count = m_Track[0].Info.Count;
				mipt.Field = m_Track[0].Info.Field;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				prop = new double[6,6] {{1,-projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,-projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,-projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
				mipt.Count = m_Track[n-1].Info.Count;
				mipt.Field = m_Track[n-1].Info.Field;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Intercept.Z = tvec[4];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(ProjectionDirection direct, double projection_Z)
		{
			double[] tvec;
			double[,] prop;
			double projection_gap;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			if(m_Track==null) throw new Exception("Track not set!");
			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				projection_gap = projection_Z - m_Track[0].Info.Intercept.Z;
				prop = new double[6,6] {{1,projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				projection_gap =  m_Track[n-1].Info.Intercept.Z - projection_Z;
				prop = new double[6,6] {{1,-projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,-projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,-projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Intercept.Z = tvec[4];
			mipt.Slope.Z = 1;

			return mipt;
		}

		public override MIPEmulsionTrackInfo ProjectVolumeTrackAtZ(Track t, ProjectionDirection direct, double projection_Z)
		{
			double[] tvec;
			double[,] prop;
			double projection_gap;
			MIPEmulsionTrackInfo mipt = new MIPEmulsionTrackInfo();

			m_Track = t;
			SetKalmanFilter();

			if(direct == ProjectionDirection.DownStream)
			{
				kf.FilterBackward();
				projection_gap = projection_Z - m_Track[0].Info.Intercept.Z;
				prop = new double[6,6] {{1,projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[0].StateVec);
				mipt.BottomZ = projection_gap + m_Track[0].Info.BottomZ;
				mipt.TopZ = projection_gap + m_Track[0].Info.TopZ;
				mipt.AreaSum = m_Track[0].Info.AreaSum;
				mipt.Sigma = m_Track[0].Info.Sigma;
				mipt.Count = m_Track[0].Info.Count;
				mipt.Field = m_Track[0].Info.Field;
			}
			else
			{
				int n = kf.Length;
				kf.FilterForward();
				projection_gap =  m_Track[n-1].Info.Intercept.Z - projection_Z;
				prop = new double[6,6] {{1,-projection_gap,0,0,0,0},{0,1,0,0,0,0},{0,0,1,-projection_gap,0,0},{0,0,0,1,0,0},{0,0,0,0,1,-projection_gap},{0,0,0,0,0,1}};
				tvec = Matrices.Product(prop, kf[n-1].StateVec);
				mipt.BottomZ = -projection_gap + m_Track[n-1].Info.BottomZ;
				mipt.TopZ = -projection_gap + m_Track[n-1].Info.TopZ;
				mipt.AreaSum = m_Track[n-1].Info.AreaSum;
				mipt.Sigma = m_Track[n-1].Info.Sigma;
				mipt.Count = m_Track[n-1].Info.Count;
				mipt.Field = m_Track[n-1].Info.Field;
			}
			mipt.Intercept.X = tvec[0];
			mipt.Slope.X = tvec[1];
			mipt.Intercept.Y = tvec[2];
			mipt.Slope.Y = tvec[3];
			mipt.Intercept.Z = tvec[4];
			mipt.Slope.Z = 1;

			return mipt;
		}
	}
}
