using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using SySal;
using SySal.Management;
using SySal.BasicTypes;
using SySal.TotalScan;

namespace SySal.Processing.MCSAnnecy
{
    /// <summary>
    /// Configuration for MomentumEstimator.
    /// </summary>
    [Serializable]
    [XmlType("MCSAnnecy.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable//, ISerializable
    {
        /// <summary>
        /// Builds a configuration initialized with default parameters.
        /// </summary>
        public Configuration()
            : base("")
        {
            RadiationLength = 5600;

            SlopeError3D_0 = 0.0021;
            SlopeError3D_1 = 0.0054;
            SlopeError3D_2 = 0.0;

            SlopeErrorLong_0 = 0.0021;
            SlopeErrorLong_1 = 0.0093;
            SlopeErrorLong_2 = 0.0;

            SlopeErrorTransv_0 = 0.0021;
            SlopeErrorTransv_1 = 0.0;
            SlopeErrorTransv_2 = 0.0;

            IgnoreLongitudinal = false;
            IgnoreTransverse = false;

            MinEntries = 3;
        }

        /// <summary>
        /// Builds a configuration initialized with default parameters, and with the specified name.
        /// </summary>
        /// <param name="name"></param>
        public Configuration(string name)
            : base(name)
        {
            RadiationLength = 5600;

            SlopeError3D_0 = 0.0021;
            SlopeError3D_1 = 0.0054;
            SlopeError3D_2 = 0.0;

            SlopeErrorLong_0 = 0.0021;
            SlopeErrorLong_1 = 0.0093;
            SlopeErrorLong_2 = 0.0;

            SlopeErrorTransv_0 = 0.0021;
            SlopeErrorTransv_1 = 0.0;
            SlopeErrorTransv_2 = 0.0;

            IgnoreLongitudinal = false;
            IgnoreTransverse = false;

            MinEntries = 3;
        }

        /// <summary>
        /// Set to <c>true</c> to ignore transverse information.
        /// </summary>
        public bool IgnoreTransverse;

        /// <summary>
        /// Set to <c>true</c> to ignore longitudinal information.
        /// </summary>
        public bool IgnoreLongitudinal;

        /// <summary>
        /// Radiation length.
        /// </summary>
        public double RadiationLength;

        /// <summary>
        /// min number of entries in the cell to accept it for fitting (def=1).
        /// </summary>
        public int MinEntries;


        // detheta  = eDT0 *(1+ eDT1*theta0- eDT2*theta0*theta0);

        public double SlopeError3D_0;

        public double SlopeError3D_1;

        public double SlopeError3D_2;


        // dethetaX = eDTx0*(1+eDTx1*theta0-eDTx2*theta0*theta0);

        public double SlopeErrorLong_0;

        public double SlopeErrorLong_1;

        public double SlopeErrorLong_2;


        // dethetaY = eDTy0*(1+eDTy1*theta0-eDTy2*theta0*theta0);

        public double SlopeErrorTransv_0;

        public double SlopeErrorTransv_1;

        public double SlopeErrorTransv_2;


        public override object Clone()
        {
            Configuration C = new Configuration();
            C.RadiationLength = this.RadiationLength;
            C.MinEntries = this.MinEntries;
            C.SlopeError3D_0 = this.SlopeError3D_0;
            C.SlopeError3D_1 = this.SlopeError3D_1;
            C.SlopeError3D_2 = this.SlopeError3D_2;
            C.SlopeErrorLong_0 = this.SlopeErrorLong_0;
            C.SlopeErrorLong_1 = this.SlopeErrorLong_1;
            C.SlopeErrorLong_2 = this.SlopeErrorLong_2;
            C.SlopeErrorTransv_0 = this.SlopeErrorTransv_0;
            C.SlopeErrorTransv_1 = this.SlopeErrorTransv_1;
            C.SlopeErrorTransv_2 = this.SlopeErrorTransv_2;
            C.IgnoreLongitudinal = this.IgnoreLongitudinal;
            C.IgnoreTransverse = this.IgnoreTransverse;
            return C;
        }
    }

    /// <summary>
    /// Provides momentum estimation using Multiple Coulomb Scattering.
    /// </summary>
    [Serializable]
    [XmlType("MCSAnnecy.MomentumEstimator")]
    public class MomentumEstimator : IManageable, IMCSMomentumEstimator
    {
        public override string ToString()
        {
            return "Annecy MCS Algorithm";
        }

        [NonSerialized]
        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        #region IManageable Members

        /// <summary>
        /// Member field on which the Configuration property relies.
        /// </summary>
        protected Configuration C = new Configuration();
        /// <summary>
        /// The configuration of the momentum estimator. Includes operational settings (such as momentum bounds) as well the specification of the material geometry.
        /// </summary>
        [XmlElement(typeof(MCSAnnecy.Configuration))]
        public SySal.Management.Configuration Config
        {
            get
            {
                return (SySal.Management.Configuration)(C.Clone());
            }
            set
            {
                C = (SySal.Processing.MCSAnnecy.Configuration)(value.Clone());
            }
        }


        /// <summary>
        /// List of connections. It is always empty for MomentumEstimator.
        /// </summary>
        public IConnectionList Connections
        {
            get { return EmptyConnectionList; }
        }

        /// <summary>
        /// GUI editor to configure the algorithm parameters.
        /// </summary>
        /// <param name="c">the configuration to be edited.</param>
        /// <returns><c>true</c> if the configuration is accepted, <c>false</c> otherwise.</returns>
        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            EditConfigForm ec = new EditConfigForm();
            ec.C = (SySal.Processing.MCSAnnecy.Configuration)(c.Clone());
            if (ec.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                c = (SySal.Processing.MCSAnnecy.Configuration)(ec.C.Clone());
                return true;
            }
            return false;
        }

        /// <summary>
        /// Monitor enable/disable. Monitoring is currently not supported (enabling the monitor results in an exception).
        /// </summary>
        public bool MonitorEnabled
        {
            get
            {
                return false;
            }
            set
            {
                if (value != false) throw new System.Exception("This object has no monitor.");
            }
        }

        /// <summary>
        /// Member field on which the Name property relies.
        /// </summary>
        [NonSerialized]
        protected string m_Name;
        /// <summary>
        /// The name of the momentum estimator.
        /// </summary>
        public string Name
        {
            get
            {
                return (string)(m_Name.Clone());
            }
            set
            {
                m_Name = (string)(value.Clone());
            }
        }

        #endregion

        /// <summary>
        /// Computes the momentum and confidence limits using positions and slopes provided.
        /// </summary>
        /// <param name="data">the position and slopes of the track (even Z-unordered). The <c>Field</c> member is used to define the plate.</param>
        /// <returns>the momentum and confidence limits.</returns>
        public MomentumResult ProcessData(SySal.Tracking.MIPEmulsionTrackInfo[] data)
        {
            int i;
            MomentumResult result = new MomentumResult();

            System.Collections.ArrayList tr = new System.Collections.ArrayList();
            tr.AddRange(data);
            tr.Sort(new DataSorter());

            int nseg = tr.Count;
            int npl = (int)(((SySal.Tracking.MIPEmulsionTrackInfo)tr[tr.Count - 1]).Field - ((SySal.Tracking.MIPEmulsionTrackInfo)tr[0]).Field) + 1;
            if (nseg < 2) throw new Exception("PMSang - Warning! nseg < 2 (" + nseg + ")- impossible to estimate momentum!");
            if (npl < nseg) throw new Exception("PMSang - Warning! npl < nseg (" + npl + ", " + nseg + ")");

            int plmax = (int)((SySal.Tracking.MIPEmulsionTrackInfo)tr[tr.Count - 1]).Field + 1;
            if (plmax < 1 || plmax > 1000) throw new Exception("PMSang - Warning! plmax = " + plmax + " - correct the segments PID's!");

            float xmean, ymean, zmean, txmean, tymean, wmean;
            float xmean0, ymean0, zmean0, txmean0, tymean0, wmean0;
            FitTrackLine(tr, out xmean0, out ymean0, out zmean0, out txmean0, out tymean0, out wmean0);
            float tmean = (float)Math.Sqrt(txmean0 * txmean0 + tymean0 * tymean0);

            SySal.Tracking.MIPEmulsionTrackInfo aas;
            float sigmax = 0, sigmay = 0;
            for (i = 0; i < tr.Count; i++)
            {
                aas = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i];
                sigmax += (float)((txmean0 - aas.Slope.X) * (txmean0 - aas.Slope.X));
                sigmay += (float)((tymean0 - aas.Slope.Y) * (tymean0 - aas.Slope.Y));
            }
            sigmax = (float)(Math.Sqrt(sigmax / tr.Count));
            sigmay = (float)(Math.Sqrt(sigmay / tr.Count));
            for (i = 0; i < tr.Count; i++)
            {
                aas = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i];
                if (Math.Abs(aas.Slope.X - txmean0) > 3 * sigmax || Math.Abs(aas.Slope.Y - tymean0) > 3 * sigmay)
                {
                    aas.Slope.X = 0; aas.Slope.Y = 0;                    
                }
            }

            FitTrackLine(tr, out xmean0, out ymean0, out zmean0, out txmean0, out tymean0, out wmean0);

            float PHI = (float)Math.Atan2(txmean0, tymean0);
            for (i = 0; i < tr.Count; i++)
            {
                aas = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i];
                float slx = (float)(aas.Slope.Y * Math.Cos(-PHI) - aas.Slope.X * Math.Sin(-PHI));
                float sly = (float)(aas.Slope.X * Math.Cos(-PHI) + aas.Slope.Y * Math.Sin(-PHI));
                aas.Slope.X = slx;
                aas.Slope.Y = sly;
            }

            FitTrackLine(tr, out xmean, out ymean, out zmean, out txmean, out tymean, out wmean);            



            // -- start calcul --

            int minentr = C.MinEntries;               // min number of entries in the cell to accept the cell for fitting
            int stepmax = npl - 1; //npl/minentr;     // max step
            int size = stepmax + 1;       // vectors size

            int maxcell = 14;

            double[] da = new double[Math.Min(size, maxcell)];
            double[] dax = new double[Math.Min(size, maxcell)];
            double[] day = new double[Math.Min(size, maxcell)];

            int[] nentr = new int[Math.Min(size, maxcell)];
            int[] nentrx = new int[Math.Min(size, maxcell)];
            int[] nentry = new int[Math.Min(size, maxcell)];

            SySal.Tracking.MIPEmulsionTrackInfo s1, s2;
            int ist;
            for (ist = 1; ist <= stepmax; ist++)         // cycle by the step size
            {
                int i1;
                for (i1 = 0; i1 < nseg - 1; i1++)          // cycle by the first seg
                {
                    s1 = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i1];
                    /* ???? KRYSS: how could this be null ???? if(!s1) continue; */
                    int i2;
                    for (i2 = i1 + 1; i2 < nseg; i2++)      // cycle by the second seg
                    {
                        s2 = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i2];
                        /* ???? KRYSS: how could this be null ???? if(!s2) continue; */
                        int icell = (int)Math.Abs(s2.Field - s1.Field);                        
                        if (icell > maxcell) continue;
                        if (icell == ist)
                        {
                            if (s2.Slope.X != 0 && s1.Slope.X != 0)
                            {
                                dax[icell - 1] += (float)((Math.Atan(s2.Slope.X) - Math.Atan(s1.Slope.X)) * (Math.Atan(s2.Slope.X) - Math.Atan(s1.Slope.X)));
                                nentrx[icell - 1] += 1;
                            }
                            if (s2.Slope.Y != 0 && s1.Slope.Y != 0)
                            {
                                day[icell - 1] += (float)((Math.Atan(s2.Slope.Y) - Math.Atan(s1.Slope.Y)) * (Math.Atan(s2.Slope.Y) - Math.Atan(s1.Slope.Y)));
                                nentry[icell - 1] += 1;
                            }
                            if (s2.Slope.X != 0.0 && s1.Slope.X != 0.0 && s2.Slope.Y != 0.0 && s1.Slope.Y != 0.0)
                            {
                                da[icell - 1] += (float)(((Math.Atan(s2.Slope.X) - Math.Atan(s1.Slope.X)) * (Math.Atan(s2.Slope.X) - Math.Atan(s1.Slope.X)))
                                    + ((Math.Atan(s2.Slope.Y) - Math.Atan(s1.Slope.Y)) * (Math.Atan(s2.Slope.Y) - Math.Atan(s1.Slope.Y))));
                                nentr[icell - 1] += 1;
                            }
                        }
                    }
                }
            }

            if (m_DiffLog != null)
            {
                m_DiffLog.WriteLine("Entries: " + da.Length);
                int u;
                m_DiffLog.WriteLine("3D");
                for (u = 0; u < nentr.Length; u++)
                    m_DiffLog.WriteLine(u + " " + nentr[u] + " " + da[u]);
                m_DiffLog.WriteLine("Longitudinal");
                for (u = 0; u < nentrx.Length; u++)
                    m_DiffLog.WriteLine(u + " " + nentrx[u] + " " + dax[u]);
                m_DiffLog.WriteLine("Transverse");
                for (u = 0; u < nentry.Length; u++)
                    m_DiffLog.WriteLine(u + " " + nentry[u] + " " + day[u]);
                m_DiffLog.Flush();
            }

            float Zcorr = (float)Math.Sqrt(1 + txmean0 * txmean0 + tymean0 * tymean0);  // correction due to non-zero track angle and crossed lead thickness

            int maxX = 0, maxY = 0, max3D = 0;                                  // maximum value for the function fit
            
            double[][] vindx = new double[Math.Min(size, maxcell)][];
            double[][] errvindx = new double[Math.Min(size, maxcell)][];
            double[][] vindy = new double[Math.Min(size, maxcell)][];
            double[][] errvindy = new double[Math.Min(size, maxcell)][];
            double[][] vind3d = new double[Math.Min(size, maxcell)][];
            double[][] errvind3d = new double[Math.Min(size, maxcell)][];
            double[] errda = new double[Math.Min(size, maxcell)];
            double[] errdax = new double[Math.Min(size, maxcell)];
            double[] errday = new double[Math.Min(size, maxcell)];

            System.Collections.ArrayList ar_vindx = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errvindx = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_vindy = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errvindy = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_vind3d = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errvind3d = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_da = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_dax = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_day = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errda = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errdax = new System.Collections.ArrayList();
            System.Collections.ArrayList ar_errday = new System.Collections.ArrayList();



            ist = 0;
            int ist1 = 0, ist2 = 0;                          // use the counter for case of missing cells 
            for (i = 0; i < vind3d.Length /* size */; i++)
            {
                if (nentrx[i] >= minentr && Math.Abs(dax[i]) < 0.1)
                {
                    if (ist >= vindx.Length) continue;

                    ar_vindx.Add(new double[1] { i + 1 });
                    ar_errvindx.Add(new double[1] { .25 });
                    ar_dax.Add(Math.Sqrt(dax[i] / (nentrx[i] * Zcorr)));
                    ar_errdax.Add((double)ar_dax[ar_dax.Count - 1] / Math.Sqrt(2 * nentrx[i]));                    
                    ist++;
                    maxX = ist;
                }
                if (nentry[i] >= minentr && Math.Abs(day[i]) < 0.1)
                {
                    if (ist1 >= vindy.Length) continue;

                    ar_vindy.Add(new double[1] { i + 1 });
                    ar_errvindy.Add(new double[1] { .25 });
                    ar_day.Add(Math.Sqrt(day[i] / (nentry[i] * Zcorr)));
                    ar_errday.Add((double)ar_day[ar_day.Count - 1] / Math.Sqrt(2 * nentry[i]));
                    ist1++;
                    maxY = ist1;
                }
                if (nentr[i] >= minentr / 2 && Math.Abs(da[i]) < 0.1)
                {
                    if (ist2 >= vind3d.Length) continue;   
                 
                    ar_vind3d.Add(new double[1] { i + 1 });                    
                    ar_errvind3d.Add(new double[1] { .25 });
                    ar_da.Add(Math.Sqrt(da[i] / (2 * nentr[i] * Zcorr)));
                    ar_errda.Add((double)ar_da[ar_da.Count - 1] / Math.Sqrt(4 * nentr[i]));
                    ist2++;
                    max3D = ist2;
                }
            }

            vindx = (double[][])ar_vindx.ToArray(typeof(double[]));
            vindy = (double[][])ar_vindy.ToArray(typeof(double[]));
            vind3d = (double[][])ar_vind3d.ToArray(typeof(double[]));
            errvindx = (double[][])ar_errvindx.ToArray(typeof(double[]));
            errvindy = (double[][])ar_errvindy.ToArray(typeof(double[]));
            errvind3d = (double[][])ar_errvind3d.ToArray(typeof(double[]));
            da = (double[])ar_da.ToArray(typeof(double));
            dax = (double[])ar_dax.ToArray(typeof(double));
            day = (double[])ar_day.ToArray(typeof(double));
            errda = (double[])ar_errda.ToArray(typeof(double));
            errdax = (double[])ar_errdax.ToArray(typeof(double));
            errday = (double[])ar_errday.ToArray(typeof(double));

            float dt = (float)(C.SlopeError3D_0 + C.SlopeError3D_1 * Math.Abs(tmean) + C.SlopeError3D_2 * tmean * tmean);  // measurements errors parametrization
            dt *= dt;
            float dtx = (float)(C.SlopeErrorLong_0 + C.SlopeErrorLong_1 * Math.Abs(txmean) + C.SlopeErrorLong_2 * txmean * txmean);  // measurements errors parametrization
            dtx *= dtx;
            float dty = (float)(C.SlopeErrorTransv_0 + C.SlopeErrorTransv_1 * Math.Abs(tymean) + C.SlopeErrorTransv_2 * tymean * tymean);  // measurements errors parametrization
            dty *= dty;

            float x0 = (float)(C.RadiationLength / 1000);


            // the fit results
            float ePx = 0.0f, ePy = 0.0f;             // the estimated momentum
            float eDPx = 0.0f, eDPy = 0.0f;           // the fit error

            float ePXmin = 0.0f, ePXmax = 0.0f;      // momentum 90% errors range
            float ePYmin = 0.0f, ePYmax = 0.0f;      // momentum 90% errors range

            // the output of PMSang
            float eP = 0.0f, eDP = 0.0f;
            float ePmin = 0.0f, ePmax = 0.0f;         // momentum 90% errors range

            /*
  eF1X = MCSErrorFunction("eF1X",x0,dtx);    eF1X->SetRange(0,14);
  eF1X->SetParameter(0,2000.);                             // starting value for momentum in GeV
  eF1Y = MCSErrorFunction("eF1Y",x0,dty);    eF1Y->SetRange(0,14);
  eF1Y->SetParameter(0,2000.);                             // starting value for momentum in GeV
  eF1 = MCSErrorFunction("eF1",x0,dt);     eF1->SetRange(0,14);
  eF1->SetParameter(0,2000.);                             // starting value for momentum in GeV
             */

            NumericalTools.AdvancedFitting.LeastSquares LSF = new NumericalTools.AdvancedFitting.LeastSquares();
            LSF.Logger = m_FitLog;
            float chi2_3D = -1.0f;
            float chi2_T = -1.0f;
            float chi2_L = -1.0f;
            if (max3D > 0)
            {
                try
                {
                    LSF.Fit(new MyNF(x0, dt), 1, vind3d, da, errvind3d, errda, 100);
                    eP = (float)(1.0f / 1000.0f * Math.Abs(LSF.BestFit[0]));
                    eDP = (float)(1.0f / 1000.0f * LSF.StandardErrors[0]);
                    EstimateMomentumError(eP, npl, tymean, out ePmin, out ePmax);
                    chi2_3D = (float)LSF.EstimatedVariance;
                }
                catch (Exception)
                {
                    ePmin = ePmax = eP = -99;
                }                                
            }
            if (maxX > 0)
            {
                try
                {
                    LSF.Fit(new MyNF(x0, dtx), 1, vindx, dax, errvindx, errdax, 100);
                    ePx = (float)(1.0f / 1000.0f * Math.Abs(LSF.BestFit[0]));
                    eDPx = (float)(1.0f / 1000.0f * LSF.StandardErrors[0]);
                    EstimateMomentumError(ePx, npl, txmean, out ePXmin, out ePXmax);
                    chi2_L = (float)LSF.EstimatedVariance;
                }
                catch (Exception)
                {
                    ePXmin = ePXmax = ePx = -99;
                }                
            }
            if (maxY > 0)
            {
                try
                {
                    LSF.Fit(new MyNF(x0, dty), 1, vindy, day, errvindy, errday, 100);
                    ePy = (float)(1.0f / 1000.0f * Math.Abs(LSF.BestFit[0]));
                    eDPy = (float)(1.0f / 1000.0f * LSF.StandardErrors[0]);
                    EstimateMomentumError(ePy, npl, tmean, out ePYmin, out ePYmax);
                    chi2_T = (float)LSF.EstimatedVariance;
                }
                catch (Exception)
                {
                    ePYmin = ePYmax = ePy = -99;
                }                
            }

            result.ConfidenceLevel = 0.90;
            if (!C.IgnoreLongitudinal && !C.IgnoreTransverse)
            {
                result.Value = Math.Round(eP / 0.01) * 0.01;
                result.LowerBound = Math.Round(ePmin / 0.01) * 0.01;
                result.UpperBound = Math.Round(ePmax / 0.01) * 0.01;
                if (tmean > 0.2 && ((chi2_T >= 0.0 && chi2_T < chi2_3D) || (chi2_3D < 0.0 && chi2_T >= 0.0)))
                {
                    result.Value = Math.Round(ePy / 0.01) * 0.01;
                    result.LowerBound = Math.Round(ePYmin / 0.01) * 0.01;
                    result.UpperBound = Math.Round(ePYmax / 0.01) * 0.01;
                }
            }
            else if (!C.IgnoreTransverse && C.IgnoreLongitudinal)
            {
                result.Value = Math.Round(ePy / 0.01) * 0.01;
                result.LowerBound = Math.Round(ePYmin / 0.01) * 0.01;
                result.UpperBound = Math.Round(ePYmax / 0.01) * 0.01;
            }
            else if (!C.IgnoreLongitudinal && C.IgnoreTransverse)
            {
                result.Value = Math.Round(ePx / 0.01) * 0.01;
                result.LowerBound = Math.Round(ePXmin / 0.01) * 0.01;
                result.UpperBound = Math.Round(ePXmax / 0.01) * 0.01;
            }
            else
            {
                result.Value = result.LowerBound = result.UpperBound = -99;
                throw new Exception("Both projections are disabled in scattering estimation!");
            }
            return result;
        }

        /// <summary>
        /// Property backer for <c>DiffDump</c>.
        /// </summary>
        protected System.IO.TextWriter m_DiffLog;
        /// <summary>
        /// The stream used to log the slope differences. Set to <c>null</c> to disable logging.
        /// </summary>
        public System.IO.TextWriter DiffLog
        {
            set { m_DiffLog = value; }
        }

        /// <summary>
        /// Property backer for <c>FitLog</c>.
        /// </summary>
        protected System.IO.TextWriter m_FitLog;
        /// <summary>
        /// The stream used to log the fit procedure. Set to <c>null</c> to disable logging.
        /// </summary>
        public System.IO.TextWriter FitLog
        {
            set { m_FitLog = value; }
        }

        internal class MCSErrorFunction
        {
            public double x0;
            public double dtx;
            public double x;
            public double Evaluate(double p)
            {
                return Math.Sqrt(214.3296 * x / x0 * Math.Pow(1 + 0.038 * Math.Log(x / x0), 2.0) / Math.Pow(p, 2.0) + dtx);
            }
        }

        private static void EstimateMomentumError(float P, int npl, float ang, out float pmin, out float pmax)
        {
            float pinv = 1.0f / P;
            float DP = (float)Mat(P, npl, ang);
            float pinvmin = (float)(pinv * (1 - DP * 1.64));
            float pinvmax = (float)(pinv * (1 + DP * 1.64));
            pmin = (1.0f / pinvmax);   //90%CL minimum momentum
            pmax = (1.0f / pinvmin);   //90%CL maximum momentum
            if (P > 1000.0f) pmax = 10000000.0f;
        }

        private static double Mat(float P, int npl, float ang)
        {
            // These parametrisations where set on 2012-02-03
            double DP = 0.0;
            if (Math.Abs(ang) < 0.2) DP = ((0.397 + 0.019 * P) / Math.Sqrt(npl) + (0.176 + 0.042 * P) + (-0.014 - 0.003 * P) * Math.Sqrt(npl));
            if (Math.Abs(ang) >= 0.2) DP = ((1.400 - 0.022 * P) / Math.Sqrt(npl) + (-0.040 + 0.051 * P) + (0.003 - 0.004 * P) * Math.Sqrt(npl));
            if (DP > 0.80) DP = 0.80;
            return DP;
        }

        private class DataSorter : System.Collections.IComparer
        {

            #region IComparer Members

            public int Compare(object x, object y)
            {
                return (int)(((SySal.Tracking.MIPEmulsionTrackInfo)x).Field - ((SySal.Tracking.MIPEmulsionTrackInfo)y).Field);
            }

            #endregion
        }

        private static int FitTrackLine(System.Collections.ArrayList tr, out float x, out float y, out float z, out float tx, out float ty, out float w)
        {
            // track fit by averaging of segments parameters and return the mean values

            int nseg = tr.Count;
            x = 0; y = 0; z = 0; tx = 0; ty = 0; w = 0;
            SySal.Tracking.MIPEmulsionTrackInfo seg = null;
            for (int i = 0; i < nseg; i++)
            {
                seg = (SySal.Tracking.MIPEmulsionTrackInfo)tr[i];
                x += (float)seg.Intercept.X;
                y += (float)seg.Intercept.Y;
                z += (float)seg.Intercept.Z;
                tx += (float)seg.Slope.X;
                ty += (float)seg.Slope.Y;
                w += (float)seg.Count;
            }
            x /= nseg;
            y /= nseg;
            z /= nseg;
            tx /= nseg;
            ty /= nseg;
            return nseg;
        }

    }    

    class MyNF : NumericalTools.Minimization.ITargetFunction
    {

        protected double m_DT;

        protected double m_X0;

        public MyNF(double x0, double dt)
        {
            m_X0 = x0;
            m_DT = dt;
        }

        #region ITargetFunction Members

        public int CountParams
        {
            get { return 2; }
        }

        public NumericalTools.Minimization.ITargetFunction Derive(int i)
        {
            switch (i)
            {
                case 0: return new MyNDP(m_X0, m_DT);
                case 1: return new MyNDX(m_X0, m_DT);
                default: throw new Exception("Invalid index");
            };
        }

        public double Evaluate(params double[] x)
        {
            if (x.Length != 2) throw new Exception("2 parameters must be passed.");
            return Math.Sqrt(214.3296 / (x[0] * x[0]) * x[1] / m_X0 * Math.Pow(1.0 + 0.038 * Math.Log(x[1] / m_X0), 2.0) + m_DT);
        }

        public double RangeMax(int i)
        {
            switch (i)
            {
                case 0: return double.PositiveInfinity;
                case 1: return 14.0;
            }
            throw new Exception("Invalid index " + i);
        }

        public double RangeMin(int i)
        {
            switch (i)
            {
                case 0: return 0.0;
                case 1: return 0.0;
            }
            throw new Exception("Invalid index " + i);
        }

        public double[] Start
        {
            get { return new double[1] { 2000.0 }; }
        }

        public bool StopMinimization(double fval, double fchange, double xchange)
        {
            return (Math.Abs(xchange) < 1e-6);
        }

        #endregion
    }

    class MyNDP : NumericalTools.Minimization.ITargetFunction
    {
        protected double m_DTx;

        protected double m_X0;

        public MyNDP(double x0, double dtx2)
        {
            m_X0 = x0;
            m_DTx = dtx2;
        }

        #region ITargetFunction Members

        public int CountParams
        {
            get { return 2; }
        }

        public NumericalTools.Minimization.ITargetFunction Derive(int i)
        {
            throw new Exception("Cannot derive");
        }

        public double Evaluate(params double[] x)
        {
            return -214.3296 * x[1] / m_X0 * (Math.Pow(1.0 + 0.038 * Math.Log(x[1] / m_X0), 2.0)) / ((x[0] * x[0] * x[0]) * Math.Sqrt(214.3296 / (x[0] * x[0]) * x[1] / m_X0 * Math.Pow(1.0 + 0.038 * Math.Log(x[1] / m_X0), 2.0) + m_DTx));
        }

        public double RangeMax(int i)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public double RangeMin(int i)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public double[] Start
        {
            get { throw new Exception("The method or operation is not implemented."); }
        }

        public bool StopMinimization(double fval, double fchange, double xchange)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion
    }

    class MyNDX : NumericalTools.Minimization.ITargetFunction
    {
        protected double m_DTx;

        protected double m_X0;

        public MyNDX(double x0, double dtx2)
        {
            m_X0 = x0;
            m_DTx = dtx2;
        }

        #region ITargetFunction Members

        public int CountParams
        {
            get { return 2; }
        }

        public NumericalTools.Minimization.ITargetFunction Derive(int i)
        {
            throw new Exception("Cannot derive");
        }

        public double Evaluate(params double[] x)
        {            
            return 0.5 / Math.Sqrt(214.3296 / (x[0] * x[0]) * x[1] / m_X0 * Math.Pow(1.0 + 0.038 * Math.Log(x[1] / m_X0), 2.0) + m_DTx) *
                (214.3296 / (x[0] * x[0]) / m_X0 * ((1.0 + 0.038 * Math.Log(x[1] / m_X0))) * ((1.0 + 0.038 * Math.Log(x[1] / m_X0)) + 2.0 * 0.038));
        }

        public double RangeMax(int i)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public double RangeMin(int i)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public double[] Start
        {
            get { throw new Exception("The method or operation is not implemented."); }
        }

        public bool StopMinimization(double fval, double fchange, double xchange)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        #endregion
    }

}