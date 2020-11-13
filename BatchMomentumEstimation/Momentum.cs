#define _LIKELIHOOD_USE_LOG_
#define _USE_MEAN_
//#define _USE_LOG_
#define _USE_G4_
//#define _USE_HEURISTIC_

using System;
using System.Collections;
using System.Text;
using System.IO;
using SySal;
using SySal.Management;
using System.Runtime.Serialization;
using System.Xml.Serialization;
using SySal.TotalScan;

namespace SySal.Processing.MCSLikelihood
{
    internal class Difference
    {
        public double slopeDiff;
        public double dZeta;
        public int first;
        public int second;

    }

    internal struct scattDifference
    {
        public double value;
        public int first;
        public int second;
    };

    internal struct ProbMomentum
    {
        public double momValue;
        public double probTot;

    };

    internal struct elMatrix
    {
        public double pureRes;
        public double errorRes;
    };


    internal struct limit
    {
        public double lwLimit;
        public double upLimit;
        public int lwIndex;
        public int upIndex;
    }


    /// <summary>
    /// Configuration for MomentumEstimator.
    /// </summary>
    [Serializable]
    [XmlType("MCSLikelihood.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable//, ISerializable
    {
        /// <summary>
        /// Builds an unitialized configuration.
        /// </summary>
        public Configuration() : base("") { }

        /// <summary>
        /// Builds a configuration with the specified name.
        /// </summary>
        /// <param name="name"></param>
        public Configuration(string name) : base(name) { }

        /// <summary>
        /// The Confidence Level for which bounds are computed.
        /// </summary>
        public double ConfidenceLevel;
        /// <summary>
        /// The measurement error on slopes (to be applied separately on each projection).
        /// </summary>
        public double SlopeError;
        /// <summary>
        /// Minimum value of the momentum to be considered.
        /// </summary>
        public double MinimumMomentum;
        /// <summary>
        /// Maximum value of the momentum to be considered.
        /// </summary>
        public double MaximumMomentum;
        /// <summary>
        /// The spanning step of the momentum interval defined by <see cref="MinimumMomentum"/> and <see cref="MaximumMomentum"/>. Output and bounds will not be more fine-grained than this value.
        /// </summary>
        public double MomentumStep;
        /// <summary>
        /// Minimum number of radiation lengths to use when building scattering angles. It need not be an integer number, and is a generalization of the concept of the measurement cell. Set to <c>0</c> to let the algorithm auto-adjust it to the best value.
        /// </summary>
        /// <remarks>Higher values of this parameter (e.g. 4 or more) yield access to higher momenta, but reduce the number of usable measurements, thus increasing fluctuations. In practice, the maximum measurable momentum is approximately given in GeV/c by <c>(0.0136/SlopeError) Sqrt(MinimumRadiationLengths)</c>.</remarks>
        public double MinimumRadiationLengths;
        /// <summary>
        /// The geometry of the scattering volumes.
        /// </summary>
        public Geometry Geometry;
        /// <summary>
        /// Yields a copy of this object.
        /// </summary>
        /// <returns>the cloned object.</returns>
        public override object Clone()
        {
            Configuration C = new Configuration();
            C.ConfidenceLevel = ConfidenceLevel;
            C.MaximumMomentum = MaximumMomentum;
            C.MinimumMomentum = MinimumMomentum;
            C.MinimumRadiationLengths = MinimumRadiationLengths;
            C.MomentumStep = MomentumStep;
            C.SlopeError = SlopeError;
            C.Geometry = new Geometry();
            C.Geometry.Layers = new Geometry.LayerStart[Geometry.Layers.Length];
            int i;
            for (i = 0; i < C.Geometry.Layers.Length; i++)
                C.Geometry.Layers[i] = Geometry.Layers[i];
            return C;
        }
    }

    /// <summary>
    /// Provides momentum estimation using Multiple Coulomb Scattering.
    /// </summary>
    [Serializable]
    [XmlType("MCSLikelihood.MomentumEstimator")]    
    public class MomentumEstimator: IManageable, IMCSMomentumEstimator
    {
        public override string ToString()
        {
            return "Likelihood MCS Algorithm";
        }

        [NonSerialized]
        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        /// <summary>
        /// Builds a new MomentumEstimator and initializes its configuration to default values.
        /// </summary>
        public MomentumEstimator()
        {
            C = new Configuration();
            C.ConfidenceLevel = 0.90;
            C.MaximumMomentum = 100.0;
            C.MinimumMomentum = 0.05;
            C.SlopeError = 0.00167;
            C.MomentumStep = 0.05;
            C.MinimumRadiationLengths = 0.0;
            C.Name = "Default momentum estimator for OPERA ECC.";
            C.Geometry = new Geometry();
            C.Geometry.Layers = new Geometry.LayerStart[116];
            int i;
            for (i = 0; i < C.Geometry.Layers.Length; i += 2)
            {                
                C.Geometry.Layers[i] = new Geometry.LayerStart();
                C.Geometry.Layers[i].RadiationLength = 29000.0;
                C.Geometry.Layers[i].ZMin = (i / 2 - 57) * 1300.0;
                C.Geometry.Layers[i].Plate = (57 - i / 2);
                C.Geometry.Layers[i + 1] = new Geometry.LayerStart();
                C.Geometry.Layers[i + 1].RadiationLength = 5600.0;
                C.Geometry.Layers[i + 1].ZMin = (i / 2 - 57) * 1300.0 + 300.0;
                C.Geometry.Layers[i + 1].Plate = 0;
            }
        }

        /// <summary>
        /// Member field on which the <see cref="AngularDiffDumpFile"/> property relies.
        /// </summary>
        [NonSerialized]
        protected string m_angularDiffDumpFile = null;
        /// <summary>
        /// Member field on which the <see cref="TrackingDumpFile"/> property relies.
        /// </summary>
        [NonSerialized]
        protected string m_tkDumpFile = null;
        /// <summary>
        /// Member field on which the <see cref="CovarianceDumpFile"/> property relies.
        /// </summary>
        [NonSerialized]
        protected string m_cvDumpFile = null;
        /// <summary>
        /// Member field on which the <see cref="LikelihoodDumpFile"/> property relies.
        /// </summary>
        [NonSerialized]
        protected string m_lkDumpFile = null;
        
        internal double myChi = -10;

        /// <summary>
        /// Sets the name of the file to which the angular difference information is dumped. Set to <c>null</c> to disable dumping. It is useful to check that the distribution of scattering data is as expected.
        /// </summary>
        [XmlIgnore]
        public string AngularDiffDumpFile
        {
            set
            {
                m_angularDiffDumpFile = value;
            }
        }

        /// <summary>
        /// Sets the name of the file to which the tracking information is dumped. Set to <c>null</c> to disable dumping. It is useful to check how elements of the covariance matrix are built.
        /// </summary>
        [XmlIgnore]
        public string TrackingDumpFile
        {
            set
            {
                m_tkDumpFile = value;
            }
        }

        /// <summary>
        /// Sets the name of the file to which the covariance matrix is dumped. Set to <c>null</c> to disable dumping. It is useful to check the covariance matrix are built.
        /// </summary>
        [XmlIgnore]
        public string CovarianceDumpFile
        {
            set
            {
                m_cvDumpFile = value;
            }
        }

        /// <summary>
        /// Sets the name of the file to which the likelihood function is dumped. Set to <c>null</c> to disable dumping. It is useful to check the shape of the curve.
        /// </summary>
        [XmlIgnore]
        public string LikelihoodDumpFile
        {
            set
            {
                m_lkDumpFile = value;
            }
        }

        internal class order : System.Collections.IComparer
        {
            public int Compare(object x, object y)
            {
                double c = ((SySal.Tracking.MIPEmulsionTrackInfo)x).Intercept.Z - ((SySal.Tracking.MIPEmulsionTrackInfo)y).Intercept.Z;
                if (c > 0.0) return 1;
                if (c < 0.0) return -1;
                return 0;
            }
        }

        /// <summary>
        /// Computes the momentum and confidence limits using positions and slopes provided.
        /// </summary>
        /// <param name="data">the position and slopes of the track (even Z-unordered).</param>
        /// <returns>the momentum and confidence limits.</returns>
        public MomentumResult ProcessData(SySal.Tracking.MIPEmulsionTrackInfo[] data)
        {
            NumericalTools.Likelihood lk = null;
           return ProcessData(data, out lk);
        }

        /// <summary>
        /// Computes the momentum and confidence limits using positions and slopes provided.
        /// </summary>
        /// <param name="data">the position and slopes of the track (even Z-unordered).</param>
        /// <param name="likelihood">the output variable that will contain the likelihood function.</param>
        /// <returns>the momentum and confidence limits.</returns>
        public MomentumResult ProcessData(SySal.Tracking.MIPEmulsionTrackInfo[] data, out NumericalTools.Likelihood likelihood)
        {
            MomentumResult myResult = new MomentumResult();
            //ordiniamo gli elementi della geometria e dei dati inseriti per z
            Geometry gm = C.Geometry;
            data = orderElementData(data);

            //Controllo che la traccia sia contenuta in tutto il brick
            if (data[0].Intercept.Z < gm.Layers[0].ZMin || data[data.Length - 1].Intercept.Z > gm.Layers[gm.Layers.Length - 1].ZMin)
            {
                throw new Exception("Data span a Z interval that exceeds the geometry bounds.");
            }

            //Calcolo gli scattering ad ogni step e li scrivo nel file di dump se c'è!!
            scattDifference[] scatDif = calcScattDiff(gm, data);
            if (m_tkDumpFile != null)
            {                
                writeInFileTK(scatDif);
            }

            //calcolo le differenze
            ArrayList resultDiff = new ArrayList();
            ArrayList resultDiffL = new ArrayList();
            double myRadiationLength = C.MinimumRadiationLengths;
            resultDiff = angularDifference(data, scatDif, myRadiationLength);
            resultDiffL = angularDifferenceL(data, scatDif, myRadiationLength);
            if (resultDiff.Count == 0)
            {
                throw new Exception("No slope difference available for computation.");
            }
            if (m_angularDiffDumpFile != null)
            {
                writeInFileAD(resultDiff);
            }

            //Calcolo elementi della matrice di covarianza e li scrivo nel file di dump se c'è!!
            ArrayList matrixFinal = new ArrayList();
            matrixFinal = covMatrix(resultDiff, scatDif);
            if (m_cvDumpFile != null)
            {
                writeInFileCV(matrixFinal);
            }

            //Calcolo la probabilità totale e li scrivo nel file di dump se c'è!!
            int maxIndex = 0;
            ProbMomentum[] finalData = totalProbability(resultDiff, matrixFinal, C.MinimumMomentum, C.MaximumMomentum, C.MomentumStep, ref maxIndex);
            if (m_lkDumpFile != null)
            {
                writeInFileLK(finalData);
            }

            //Calcolo i limiti di confidenza
            double maxLike = finalData[maxIndex].probTot;
            double myCutLike;
            myCutLike = cutLike(maxLike);
            limit myLimit;
            myLimit = limitCalculation(finalData, myCutLike, maxIndex);

            //Scrivo i risultati
            myResult.Value = finalData[maxIndex].momValue;
            myResult.LowerBound = myLimit.lwLimit;
            myResult.UpperBound = myLimit.upLimit;
            myResult.ConfidenceLevel = 0.90;

            double[] lkval = new double[finalData.Length];
            int i;
            for (i = 0; i < lkval.Length; i++)
                lkval[i] = finalData[i].probTot;
            likelihood = new NumericalTools.OneParamLogLikelihood(finalData[0].momValue, finalData[finalData.Length - 1].momValue, lkval, "P");
            return myResult;
            /*
            finalData = totalProbability(resultDiffL, matrixFinal, C.MinimumMomentum, C.MaximumMomentum, C.MomentumStep, ref maxIndex);
            for (i = 0; i < lkval.Length; i++)
                lkval[i] = finalData[i].probTot;
            likelihood = new NumericalTools.OneParamLogLikelihood(C.MomentumStep, likelihood, new NumericalTools.OneParamLogLikelihood(finalData[0].momValue, finalData[finalData.Length - 1].momValue, lkval, "P"));
            myResult.Value = likelihood.Best(0);
            double[] cr = likelihood.ConfidenceRegions(0, C.ConfidenceLevel);
            myResult.LowerBound = cr[0];
            myResult.UpperBound = cr[1];
            return myResult;
             */
        }

        private Geometry orderElementGeometry(Geometry toOrder)
        {
            Geometry myGeometry = new Geometry();
            ArrayList geom = new ArrayList();
            int i;
            for (i = 0; i < toOrder.Layers.Length; i++)
            {
                Geometry.LayerStart info = new Geometry.LayerStart();
                info = toOrder.Layers[i];
                geom.Add(info);
            }
            IComparer myComparer = new Geometry.order();
            geom.Sort(myComparer);

            Geometry.LayerStart[] gmArray = (Geometry.LayerStart[])geom.ToArray(typeof(Geometry.LayerStart));
            myGeometry.Layers = gmArray;
            return myGeometry;
        }

        private SySal.Tracking.MIPEmulsionTrackInfo[] orderElementData(SySal.Tracking.MIPEmulsionTrackInfo[] toOrder)
        {
            SySal.Tracking.MIPEmulsionTrackInfo[] myData = new SySal.Tracking.MIPEmulsionTrackInfo[toOrder.Length];
            ArrayList data = new ArrayList();
            int i;
            for (i = 0; i < toOrder.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                info = toOrder[i];
                data.Add(info);
            }

            IComparer myComparer = new MomentumEstimator.order();
            data.Sort(myComparer);

            SySal.Tracking.MIPEmulsionTrackInfo[] tkArray = (SySal.Tracking.MIPEmulsionTrackInfo[])data.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
            myData = tkArray;
            return myData;
        }

        private SySal.Tracking.MIPEmulsionTrackInfo Extrapolate(SySal.Tracking.MIPEmulsionTrackInfo data1, SySal.Tracking.MIPEmulsionTrackInfo data2, double z)
        {
            SySal.Tracking.MIPEmulsionTrackInfo Result = new SySal.Tracking.MIPEmulsionTrackInfo();
            double mult = (data2.Intercept.X - data1.Intercept.X) / (data2.Intercept.Z - data1.Intercept.Z);
            Result.Intercept.X = data1.Intercept.X + (z - data1.Intercept.Z) * mult;
            mult = (data2.Intercept.Y - data1.Intercept.Y) / (data2.Intercept.Z - data1.Intercept.Z);
            Result.Intercept.Y = data1.Intercept.Y + (z - data1.Intercept.Z) * mult;
            Result.Intercept.Z = z;
            return Result;


        }

        private double calculateSingleScatt(SySal.Tracking.MIPEmulsionTrackInfo data1, SySal.Tracking.MIPEmulsionTrackInfo data2, double X0)
        {
            //Calcolo solo DL/XO mi devo ricordare di moltiplicare per 13.6 al quadrato e dividere per p^2 e fare la radice
            double result;
            double dx = data1.Intercept.X - data2.Intercept.X;
            double dy = data1.Intercept.Y - data2.Intercept.Y;
            double dz = data1.Intercept.Z - data2.Intercept.Z;
            result = Math.Sqrt(dx * dx + dy * dy + dz * dz) / X0;
#if _USE_LOG_    
#if _USE_G4_
            double lg = 1.0 + 0.038 * Math.Log(result);
            result *= (lg * lg);
#else
            double lg = 1.0 + 0.038 * Math.Log(result);
            result *= (lg * lg);
#endif
#endif

            return result;
        }
        
        private scattDifference[] calcScattDiff(Geometry gm, SySal.Tracking.MIPEmulsionTrackInfo[] data)
        {
            int i, k;
            Geometry.LayerStart[] geom = gm.Layers;
            scattDifference[] myScattering = new scattDifference[data.Length - 1];

            double value = 0.0;
            for (i = 0; i < data.Length - 1; i++)
            {
                value = 0.0;
                for (k = 0; data[i].Intercept.Z >= geom[k].ZMin; k++) ;
                if (data[i + 1].Intercept.Z > geom[k].ZMin)
                {
                    myScattering[i].first = i;
                    myScattering[i].second = i + 1;
                    SySal.Tracking.MIPEmulsionTrackInfo info2 = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info2 = data[i];
                    do
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        info = (geom[k].ZMin == data[i + 1].Intercept.Z) ? data[i + 1] : Extrapolate(data[i], data[i + 1], geom[k].ZMin);
                        value = value + calculateSingleScatt(info2, info, geom[k - 1].RadiationLength);
                        info2 = info;
                    } 
                    while ((i + 1) < data.Length - 1 && data[i + 1].Intercept.Z >= geom[++k].ZMin);

                    if (data[i + 1].Intercept.Z < geom[k].ZMin)
                    {
                        value = value + calculateSingleScatt(info2, data[i + 1], geom[k - 1].RadiationLength);
                    }
                    myScattering[i].value = value;
                }
                else
                {
                    value = calculateSingleScatt(data[i], data[i + 1], geom[k - 1].RadiationLength);
                    myScattering[i].value = value;
                    myScattering[i].first = i;
                    myScattering[i].second = i + 1;
                }                
            }
            return myScattering;


        }        

        private void writeInFileTK(scattDifference[] myData)
        {
            int i;
            StreamWriter tkData = new StreamWriter(m_tkDumpFile);


            for (i = 0; i < myData.Length; i++)
            {
                tkData.WriteLine(myData[i].first + " " + myData[i].second + " " + myData[i].value);

            }
            tkData.Flush();
            tkData.Close();
        }

        private void writeInFileAD(ArrayList myData)
        {            
            StreamWriter adData = new StreamWriter(m_angularDiffDumpFile, true);
            int i;
            for (i = 0; i < myData.Count; i++)
            {
                Difference d = (Difference)myData[i];
                adData.WriteLine(i + " " + d.first + " " + d.second + " " + d.dZeta + " " + d.slopeDiff);
            }
            adData.Flush();
            adData.Close();
        }

        private void writeInFileCV(ArrayList myData)
        {
            double[][] pureMatrix = (double[][])myData[0];
            double[][] errorMatrix = (double[][])myData[1];
            StreamWriter cvData = new StreamWriter(m_cvDumpFile);
            cvData.WriteLine("pureElement");
            int i;
            int j;
            for (i = 0; i < pureMatrix.GetLength(0); i++)
            {

                cvData.WriteLine(" ");
                for (j = 0; j <= i; j++)
                {
                    cvData.Write(pureMatrix[i][j] + " ");

                }
            }
            cvData.WriteLine(" ");
            cvData.WriteLine("errorElement");

            for (i = 0; i < errorMatrix.GetLength(0); i++)
            {
                cvData.WriteLine(" ");
                for (j = 0; j <= i; j++)
                {
                    cvData.Write(errorMatrix[i][j] + " ");

                }
            }

            cvData.Flush();
            cvData.Close();


        }

        private void writeInFileLK(ProbMomentum[] myData)
        {
            int i;
            StreamWriter lkData = new StreamWriter(m_lkDumpFile);


            for (i = 0; i < myData.Length; i++)
            {
                lkData.WriteLine(myData[i].momValue + " " + myData[i].probTot);

            }
            lkData.Flush();
            lkData.Close();

        }

        private ArrayList angularDifference(SySal.Tracking.MIPEmulsionTrackInfo[] data, scattDifference[] firstStep, double minRL)
        {
            ArrayList dataDiff = new ArrayList();
            double den = 0.0;
#if _USE_MEAN_
            /*
            double msx = (data[0].Slope.X + data[data.Length - 1].Slope.X) * 0.5;
            double msy = (data[0].Slope.Y + data[data.Length - 1].Slope.Y) * 0.5;
             */
            double msx = 0.0;
            double msy = 0.0;
            foreach (SySal.Tracking.MIPEmulsionTrackInfo info in data)
            {
                msx += info.Slope.X;
                msy += info.Slope.Y;
            }
            msx /= data.Length;
            msy /= data.Length;
            den = msx * msx + msy * msy;
            if (den <= 0.0)
            {
                msx = 0.1;
                msy = 0.1;
                den = msx * msx + msy * msy;
            }
            den = 1.0 / Math.Sqrt(den);
#endif
            //calcoliamo le differenze
            int i;
            if (minRL <= 0.0)
            {
                double sum = 0.0;
                for (i = 0; i < firstStep.Length; i++)
                    sum += firstStep[i].value;
                minRL = 0.9 * sum * 0.5;
            }
            for (i = 0; i < data.Length; i++)
            {
                double sum = 0.0;
                int j;
                for (j = i; j < firstStep.Length && sum < minRL; j++)
                    sum += firstStep[j].value;
                    
//                j--;
//                if (i + j >= data.Length) break;
                //if (j >= firstStep.Length) break;
                if (sum < minRL) break;
                Difference info = new Difference();
                double zDiffTest = Math.Abs(data[i].Intercept.Z - data[j].Intercept.Z);
                if (zDiffTest > 1.0) //per essere sicuri di non prendere deltaZ = 0
                {
#if _USE_X_
                    info.slopeDiff = data[i].Slope.X - data[j].Slope.X;
#elif _USE_Y_
                    info.slopeDiff = data[i].Slope.Y - data[j].Slope.Y;
#elif _USE_THETA_
                    const double Theta = Math.PI * 0.25;
                    info.slopeDiff = Math.Cos(Theta) * (data[i].Slope.X - data[j].Slope.X) - Math.Sin(Theta) * (data[i].Slope.Y - data[j].Slope.Y);
#elif _USE_S0_
                    den = Math.Sqrt(data[0].Slope.X * data[0].Slope.X + data[0].Slope.Y * data[0].Slope.Y);
                    info.slopeDiff = (((data[i].Slope.X - data[j].Slope.X) * (data[0].Slope.Y) - (data[i].Slope.Y - data[j].Slope.Y) * (data[0].Slope.X)) / den);
                    //Console.WriteLine(" " + info.slopeDiff);
#elif _USE_MEAN_
                    info.slopeDiff = (((data[i].Slope.X - data[j].Slope.X) * msy - (data[i].Slope.Y - data[j].Slope.Y) * msx) * den);
#else
                    den = Math.Sqrt(data[i].Slope.X * data[i].Slope.X + data[i].Slope.Y * data[i].Slope.Y);
                    info.slopeDiff = Math.Abs(((data[i].Slope.X - data[j].Slope.X) * (data[i].Slope.Y) - (data[i].Slope.Y - data[j].Slope.Y) * (data[i].Slope.X)) / den);
#endif
                    info.first = i;
                    info.second = j; //calcolo valore[i]-valore[(i+j)]]
                    info.dZeta = zDiffTest;
                    dataDiff.Add(info);
                }
            }

            return dataDiff;
        }

        private ArrayList angularDifferenceL(SySal.Tracking.MIPEmulsionTrackInfo[] data, scattDifference[] firstStep, double minRL)
        {
            ArrayList dataDiff = new ArrayList();
            double den = 0.0;
#if _USE_MEAN_
            double msx = (data[0].Slope.Y + data[data.Length - 1].Slope.Y) * 0.5;
            double msy = -(data[0].Slope.X + data[data.Length - 1].Slope.X) * 0.5;
            den = msx * msx + msy * msy;
            if (den <= 0.0)
            {
                msx = 0.1;
                msy = 0.1;
                den = msx * msx + msy * msy;
            }
            den = 1.0 / Math.Sqrt(den);
#endif
            //calcoliamo le differenze
            int i;
            if (minRL <= 0.0)
            {
                double sum = 0.0;
                for (i = 0; i < firstStep.Length; i++)
                    sum += firstStep[i].value;
                minRL = 0.9 * sum * 0.5;
            }
            for (i = 0; i < data.Length; i++)
            {
                double sum = 0.0;
                int j;
                for (j = i; j < firstStep.Length && sum < minRL; j++)
                    sum += firstStep[j].value;

                //                j--;
                //                if (i + j >= data.Length) break;
                //if (j >= firstStep.Length) break;
                if (sum < minRL) break;
                Difference info = new Difference();
                double zDiffTest = Math.Abs(data[i].Intercept.Z - data[j].Intercept.Z);
                if (zDiffTest > 1.0) //per essere sicuri di non prendere deltaZ = 0
                {
#if _USE_X_
                    info.slopeDiff = data[i].Slope.X - data[j].Slope.X;
#elif _USE_Y_
                    info.slopeDiff = data[i].Slope.Y - data[j].Slope.Y;
#elif _USE_THETA_
                    const double Theta = Math.PI * 0.25;
                    info.slopeDiff = Math.Cos(Theta) * (data[i].Slope.X - data[j].Slope.X) - Math.Sin(Theta) * (data[i].Slope.Y - data[j].Slope.Y);
#elif _USE_S0_
                    den = Math.Sqrt(data[0].Slope.X * data[0].Slope.X + data[0].Slope.Y * data[0].Slope.Y);
                    info.slopeDiff = (((data[i].Slope.X - data[j].Slope.X) * (data[0].Slope.Y) - (data[i].Slope.Y - data[j].Slope.Y) * (data[0].Slope.X)) / den);
                    //Console.WriteLine(" " + info.slopeDiff);
#elif _USE_MEAN_
                    info.slopeDiff = (((data[i].Slope.X - data[j].Slope.X) * msy - (data[i].Slope.Y - data[j].Slope.Y) * msx) * den);
#else
                    den = Math.Sqrt(data[i].Slope.X * data[i].Slope.X + data[i].Slope.Y * data[i].Slope.Y);
                    info.slopeDiff = Math.Abs(((data[i].Slope.X - data[j].Slope.X) * (data[i].Slope.Y) - (data[i].Slope.Y - data[j].Slope.Y) * (data[i].Slope.X)) / den);
#endif
                    info.first = i;
                    info.second = j; //calcolo valore[i]-valore[(i+j)]]
                    info.dZeta = zDiffTest;
                    dataDiff.Add(info);
                }
            }

            return dataDiff;
        }

        private elMatrix calculateElementMatrix(Difference data1, Difference data2, scattDifference[] firstStep)
        {
            elMatrix finalResult = new elMatrix();
            finalResult.pureRes = 0.0;
            finalResult.errorRes = 0.0;
            int i;
            int k;
            int c1 = -1;
            int c2 = -2;
            if (data1.first > data2.second || data2.first > data1.second)
            {
                finalResult.pureRes = 0.0;
                finalResult.errorRes = 0.0;
            }
            else if (data1.first == data2.second || data2.first == data1.second)
            {
                finalResult.pureRes = 0.0;
                finalResult.errorRes = - C.SlopeError * C.SlopeError;
            }
            else if (data1.first == data2.first && data1.second == data2.second)
            {
                for (i = 0; i < firstStep.Length; i++)
                {
                    if (firstStep[i].first == data1.first) c1 = i;
                    if (firstStep[i].second == data1.second) c2 = i;
                }
                if (c1 == c2) //vuol dire che lo step è uno solo
                {
                    finalResult.pureRes = firstStep[c1].value;
                    finalResult.errorRes = 2.0 * C.SlopeError * C.SlopeError;
                }
                else //vuol dire che c'è più di 1 cella ed io sommo i sigma delle celle intermedie
                {
                    for (k = c1; k <= c2; k++)
                    {
                        finalResult.pureRes = finalResult.pureRes + firstStep[k].value;
                    }
                    finalResult.errorRes = 2 * C.SlopeError * C.SlopeError;
                }
            }
            else if (data1.first == data2.first)
            {
                c1 = data1.first;
                c2 = Math.Min(data1.second, data2.second);
                for (k = c1; k < c2; k++)
                {
                    finalResult.pureRes = finalResult.pureRes + firstStep[k].value;
                }
                finalResult.errorRes = C.SlopeError * C.SlopeError;

            }
            else if (data1.second == data2.second)
            {
                c1 = Math.Max(data1.first, data1.first);
                c2 = data1.second;
                for (k = c1; k < c2; k++)
                {
                    finalResult.pureRes = finalResult.pureRes + firstStep[k].value;
                }
                finalResult.errorRes = C.SlopeError * C.SlopeError;
            }
            else
            {
                c2 = Math.Min(data1.second, data2.second);
                c1 = Math.Max(data1.first, data2.first);
                for (k = c1; k < c2; k++)
                {
                    finalResult.pureRes = finalResult.pureRes + firstStep[k].value;
                }
                finalResult.errorRes = 0;
            }
            return finalResult;

        }

        private ArrayList covMatrix(ArrayList dataDiff, scattDifference[] firstStep)
        {
            Difference[] diffArray = (Difference[])dataDiff.ToArray(typeof(Difference));
            double[][] pureMatrix = new double[dataDiff.Count][];
            double[][] errorMatrix = new double[dataDiff.Count][];
            int i;
            int j;

            for (i = 0; i < diffArray.Length; i++)
            {

                pureMatrix[i] = new double[i + 1];
                errorMatrix[i] = new double[i + 1];

                for (j = 0; j <= i; j++)
                {

                    elMatrix info = new elMatrix();
                    info = calculateElementMatrix(diffArray[i], diffArray[j], firstStep);                    
                    pureMatrix[i][j] = info.pureRes;
                    errorMatrix[i][j] = info.errorRes;

                }
            }
            ArrayList matrixFinal = new ArrayList();
            matrixFinal.Add(pureMatrix);
            matrixFinal.Add(errorMatrix);
            return matrixFinal;
        }

        private ProbMomentum[] totalProbability(ArrayList diffDataFinal, ArrayList matrixElement, double pMin, double pMax, double step, ref int maxIndex)
        {
            int arrayDim = (Int32)(((pMax - pMin) / step) + 1);
            ProbMomentum[] finalMomentumProbability = new ProbMomentum[arrayDim];
            Difference[] myDiff = (Difference[])diffDataFinal.ToArray(typeof(Difference));
            int i;
            int k;
            int j;


            for (i = 0; i <= arrayDim - 1; i++)
            {

                ProbMomentum info = new ProbMomentum();
                double actualP = (i * step) + pMin;
                info.momValue = actualP;
                double[][] invMatrix = new double[diffDataFinal.Count][];
                double det = -1;
                invMatrix = calculateInverse(matrixElement, actualP, ref det);
                double mult = 1.0 / (/*2 * Math.PI * */ Math.Sqrt(det));
                double sum = 0;
                double[] firstProd = new double[diffDataFinal.Count];

                for (k = 0; k < diffDataFinal.Count; k++)
                {
                    sum = 0;
                    for (j = 0; j < diffDataFinal.Count; j++)
                    {
                        sum = sum + ((j >= k) ? invMatrix[j][k] : invMatrix[k][j]) * myDiff[j].slopeDiff;
                    }
                    firstProd[k] = sum;
                }
                sum = 0;
                for (k = 0; k < diffDataFinal.Count; k++)
                {
                    sum = sum + firstProd[k] * myDiff[k].slopeDiff;
                }
#if _LIKELIHOOD_USE_LOG_
                info.probTot = Math.Log(mult) - 0.5 * sum;
#else
                info.probTot = mult * Math.Exp(-0.5 * sum);
#endif
                finalMomentumProbability[i] = info;
            }
            maxIndex = 0;
            for (i = 1; i <= arrayDim - 2; i++)
            {
                if (finalMomentumProbability[maxIndex].probTot < finalMomentumProbability[i].probTot)
                {
                    maxIndex = i;

                }

            }
            return finalMomentumProbability;

        }

        private double[][] calculateInverse(ArrayList matrixElement, double p, ref double det)
        {
            double[][] pureMatrix = (double[][])matrixElement[0];
            double[][] errorMatrix = (double[][])matrixElement[1];
            double[][] returnMatrix = new double[pureMatrix.GetLength(0)][];
            int i, j;
            for (i = 0; i < pureMatrix.GetLength(0); i++)
            {
                returnMatrix[i] = new double[i + 1];

                for (j = 0; j <= i; j++)
                {
#if _USE_G4_
                    returnMatrix[i][j] = (pureMatrix[i][j] * 0.01464 * 0.01464) / (p * p) + errorMatrix[i][j];
#elif _USE_HEURISTIC_
                    returnMatrix[i][j] = (pureMatrix[i][j] * 0.0162 * 0.0162) / (p * p) + errorMatrix[i][j];
#else
                    returnMatrix[i][j] = (pureMatrix[i][j] * 0.0136 * 0.0136) / (p * p) + errorMatrix[i][j];
#endif
                }
            }

            NumericalTools.Cholesky ch = new NumericalTools.Cholesky(returnMatrix, 0);
            det = ch.Determinant;
            returnMatrix = ch.Inverse(0);
            return returnMatrix;
        }
        private double cutLike(double maxLike)
        {
            double cutLk;

#if _LIKELIHOOD_USE_LOG_
            cutLk = maxLike - 0.5 * myChi;
#else
            cutLk = maxLike * Math.Exp(-0.5 * myChi);
#endif
            return cutLk;

        }

        private double SearchConf(double lvlConf)
        {
            double result = 9.0;
            int lowLimit = 0;
            int upLimit = (int)(Chi2Integ.Length * 0.5);
            int centralValue;
            bool flag = true;

            do
            {
                if ((upLimit - lowLimit) % 2 == 0)
                {
                    centralValue = (int)((upLimit - lowLimit) * 0.5);
                }
                else
                {
                    centralValue = (int)((upLimit - lowLimit - 1) * 0.5);
                }

                if (Chi2Integ[centralValue + lowLimit, 1] < lvlConf)
                {
                    if (Chi2Integ[centralValue + lowLimit + 1, 1] < lvlConf)
                    {
                        lowLimit = centralValue + lowLimit;

                    }
                    else
                    {
                        flag = false;
                        lowLimit = centralValue + lowLimit;
                        upLimit = 1 + lowLimit;
                    }

                }
                else
                {
                    if (Chi2Integ[centralValue + lowLimit - 1, 1] > lvlConf)
                    {
                        upLimit = centralValue + lowLimit;
                    }
                    else
                    {
                        flag = false;
                        lowLimit = centralValue - 1 + lowLimit;
                        upLimit = 1 + lowLimit;
                    }
                }


            } while (flag == true);

            result = Interpolation(lowLimit, upLimit, lvlConf);

            return result;

        }

        private double Interpolation(int lw, int up, double lConf)
        {
            double chiInt;
            double a;
            double b;
            double x1 = Chi2Integ[lw, 0];
            double y1 = Chi2Integ[lw, 1];
            double x2 = Chi2Integ[up, 0];
            double y2 = Chi2Integ[up, 1];
            a = (y1 - y2) / (x1 - x2);
            b = (y2 - a * x2);
            chiInt = (lConf - b) / a;
            return chiInt;


        }

        private limit limitCalculation(ProbMomentum[] myData, double cut, int maxIndex)
        {
            limit calcLimit;
            int i;
            for (i = maxIndex - 1; i > 0 && myData[i].probTot > cut; i--) ;
            calcLimit.lwLimit = myData[Math.Max(i, 0)].momValue;
            calcLimit.lwIndex = i;

            for (i = maxIndex + 1; i < myData.Length && myData[i].probTot > cut; i++) ;
            calcLimit.upLimit = myData[Math.Min(i, myData.Length - 1)].momValue;
            calcLimit.upIndex = i;

            return calcLimit;
        }

        private double[,] Chi2Integ = new double[,] 
            {{0, 0}, {0.1, 0.24817036595415062}, {0.2, 0.34527915398142295}, {0.30000000000000004, 0.4161175792296349}, {0.4, 0.47291074313446196}, {0.5, 0.5204998778130466}, {0.6000000000000001, 0.5614219739190001}, 
 {0.7000000000000001, 0.5972163057535244}, {0.8, 0.6289066304773024}, {0.9, 0.6572182888520886}, {1.0, 0.6826894921370861}, {1.1, 0.7057338956950372}, {1.2000000000000002, 0.726678321707702}, 
 {1.3, 0.7457867763960357}, {1.4000000000000001, 0.7632764293621426}, {1.5, 0.7793286380801533}, {1.6, 0.7940967892679318}, {1.7000000000000002, 0.8077120228884803}, {1.8, 0.8202875051210001}, 
 {1.9000000000000001, 0.8319216809650295}, {2.0, 0.8427007929497151}, {2.1, 0.852700861377324}, {2.2, 0.8619892624313404}, {2.3000000000000003, 0.8706260011637018}, {2.4000000000000004, 0.8786647496415179}, 
 {2.5, 0.8861537019933419}, {2.6, 0.8931362850066205}, {2.7, 0.8996517535377093}, {2.8000000000000003, 0.9057356931587897}, {2.9000000000000004, 0.9114204474202231}, {3.0, 0.9167354833364496}, 
 {3.1, 0.92170770585359}, {3.2, 0.9263617298796973}, {3.3000000000000003, 0.9307201167787981}, {3.4000000000000004, 0.93480358092187}, {3.5, 0.9386311708605979}, {3.6, 0.9422204288764029}, 
 {3.7, 0.9455875320083986}, {3.8000000000000003, 0.9487474171426306}, {3.9000000000000004, 0.9517138923233183}, {4.0, 0.9544997361036416}, {4.1000000000000005, 0.9571167864726001}, {4.2, 0.9595760206630913}, 
 {4.3, 0.9618876269547866}, {4.4, 0.9640610694259775}, {4.5, 0.9661051464753109}, {4.6000000000000005, 0.9680280438223513}, {4.7, 0.9698373826014909}, {4.800000000000001, 0.9715402630836893}, 
 {4.9, 0.9731433044924757}, {5.0, 0.9746526813225316}, {5.1000000000000005, 0.9760741565193733}, {5.2, 0.9774131118358204}, {5.300000000000001, 0.9786745756440015}, {5.4, 0.9798632484496537}, 
 {5.5, 0.9809835263276995}, {5.6000000000000005, 0.9820395224739215}, {5.7, 0.9830350870464127}, {5.800000000000001, 0.9839738254520148}, {5.9, 0.9848591152166976}, {6.0, 0.9856941215645704}, 
 {6.1000000000000005, 0.986481811817593}, {6.2, 0.9872249687169125}, {6.300000000000001, 0.987926202756877}, {6.4, 0.9885879636139983}, {6.5, 0.9892125507453295}, {6.6000000000000005, 0.9898021232237596}, 
 {6.7, 0.9903587088715023}, {6.800000000000001, 0.9908842127474916}, {6.9, 0.991380425039398}, {7.0, 0.9918490284064975}, {7.1000000000000005, 0.9922916048155798}, {7.2, 0.9927096419084641}, 
 {7.300000000000001, 0.993104538936381}, {7.4, 0.9934776122935333}, {7.5, 0.9938301006794557}, {7.6000000000000005, 0.9941631699173544}, {7.7, 0.9944779174533993}, {7.800000000000001, 0.9947753765599386}, 
 {7.9, 0.995056520263772}, {8.0, 0.9953222650189529}, {8.1, 0.9955734741420801}, {8.200000000000001, 0.9958109610266406}, {8.3, 0.9960354921517062}, {8.4, 0.9962477898991262}, {8.5, 0.996448535192294}, 
 {8.6, 0.9966383699685876}, {8.700000000000001, 0.9968178994967017}, {8.8, 0.9969876945492544}, {8.9, 0.9971482934403076}, {9.0, 0.9973002039367397}, {9.1, 0.9974439050517705}, 
 {9.200000000000001, 0.9975798487283523}, {9.3, 0.997708461419591}, {9.4, 0.9978301455728701}, {9.5, 0.9979452810238681}, {9.600000000000001, 0.9980542263062607}, {9.700000000000001, 0.9981573198824683}, 
 {9.8, 0.9982548813004711}, {9.9, 0.9983472122813637}, {10.0, 0.9984345977419977}, {10.100000000000001, 0.9985173067567874}, {10.200000000000001, 0.9985955934624687}, {10.3, 0.9986696979093534}, 
 {10.4, 0.9987398468623898}, {10.5, 0.998806254555128}, {10.600000000000001, 0.9988691233994723}, {10.700000000000001, 0.9989286446539296}, {10.8, 0.9989849990528868}, {10.9, 0.9990383573992755}, 
 {11.0, 0.9990888811228464}, {11.100000000000001, 0.9991367228061273}, {11.200000000000001, 0.9991820266800054}, {11.3, 0.9992249290907544}, {11.4, 0.9992655589402143}, {11.5, 0.9993040381007254}, 
 {11.600000000000001, 0.999340481806309}, {11.700000000000001, 0.9993749990215067}, {11.8, 0.9994076927891999}, {11.9, 0.9994386605586406}, {12.0, 0.9994679944948608}, 
 {12.100000000000001, 0.9994957817705509}, {12.200000000000001, 0.9995221048414313}, {12.3, 0.9995470417060764}, {12.4, 0.9995706661511018}, {12.5, 0.9995930479825549}, 
 {12.600000000000001, 0.9996142532443181}, {12.700000000000001, 0.9996343444242616}, {12.8, 0.9996533806488651}, {12.9, 0.9996714178669622}, {13.0, 0.9996885090232324}, 
 {13.100000000000001, 0.999704704222034}, {13.200000000000001, 0.99972005088212}, {13.3, 0.9997345938827679}, {13.4, 0.9997483757018011}, {13.5, 0.9997614365459712}, 
 {13.600000000000001, 0.9997738144741289}, {13.700000000000001, 0.9997855455135929}, {13.8, 0.9997966637701052}, {13.9, 0.999807201531726}, {14.0, 0.9998171893670182}, 
 {14.100000000000001, 0.9998266562178358}, {14.200000000000001, 0.9998356294870239}, {14.3, 0.9998441351213109}, {14.4, 0.9998521976896656}, {14.5, 0.9998598404573705}, 
 {14.600000000000001, 0.9998670854560499}, {14.700000000000001, 0.9998739535498791}, {14.8, 0.9998804644981869}, {14.9, 0.9998866370146501}, {15.0, 0.9998924888232706}, 
 {15.100000000000001, 0.9998980367113135}, {15.200000000000001, 0.9999032965793732}, {15.3, 0.9999082834887232}, {15.4, 0.9999130117061078}, {15.5, 0.9999174947461071}, 
 {15.600000000000001, 0.9999217454112145}, {15.700000000000001, 0.9999257758297487}, {15.8, 0.9999295974917254}, {15.9, 0.9999332212827912}, {16.0, 0.999936657516334}, {16.1, 0.9999399159638643}, 
 {16.2, 0.9999430058837666}, {16.3, 0.9999459360485029}, {16.400000000000002, 0.9999487147703593}, {16.5, 0.9999513499258151}, {16.6, 0.9999538489785995}, {16.7, 0.9999562190015227}, 
 {16.8, 0.9999584666971332}, {16.900000000000002, 0.999960598417275}, {17.0, 0.999962620181598}, {17.1, 0.9999645376950854}, {17.2, 0.9999663563646388}, {17.3, 0.9999680813147889}, 
 {17.400000000000002, 0.999969717402563}, {17.5, 0.9999712692315651}, {17.6, 0.999972741165306}, {17.7, 0.999974137339826}, {17.8, 0.9999754616756473}, {17.900000000000002, 0.9999767178890924}, 
 {18.0, 0.9999779095030016}, {18.1, 0.9999790398568845}, {18.2, 0.9999801121165326}, {18.3, 0.9999811292831234}, {18.400000000000002, 0.9999820942018428}, {18.5, 0.9999830095700523}, 
 {18.6, 0.9999838779450265}, {18.7, 0.9999847017512785}, {18.8, 0.9999854832875009}, {18.900000000000002, 0.9999862247331419}, {19.0, 0.9999869281546333}, {19.1, 0.9999875955112905}, 
 {19.200000000000003, 0.9999882286609025}, {19.3, 0.9999888293650284}, {19.400000000000002, 0.9999893992940136}, {19.5, 0.9999899400317455}, {19.6, 0.9999904530801548}, 
 {19.700000000000003, 0.9999909398634848}, {19.8, 0.999991401732336}, {19.900000000000002, 0.999991839967496}, {20.0, 0.9999922557835688}, {20.1, 0.9999926503324212}, 
 {20.200000000000003, 0.9999930247064369}, {20.3, 0.9999933799416093}, {20.400000000000002, 0.9999937170204702}, {20.5, 0.9999940368748627}, {20.6, 0.9999943403885724}, 
 {20.700000000000003, 0.9999946283998209}, {20.8, 0.9999949017036266}, {20.900000000000002, 0.9999951610540446}, {21.0, 0.9999954071662884}, {21.1, 0.999995640718743}, 
 {21.200000000000003, 0.9999958623548701}, {21.3, 0.9999960726850168}, {21.400000000000002, 0.9999962722881278}, {21.5, 0.9999964617133694}, {21.6, 0.9999966414816709}, 
 {21.700000000000003, 0.9999968120871814}, {21.8, 0.9999969739986567}, {21.900000000000002, 0.9999971276607668}, {22.0, 0.9999972734953442}, {22.1, 0.9999974119025575}, 
 {22.200000000000003, 0.9999975432620352}, {22.3, 0.9999976679339202}, {22.400000000000002, 0.9999977862598772}, {22.5, 0.9999978985640443}, {22.6, 0.999998005153937}, 
 {22.700000000000003, 0.9999981063213053}, {22.8, 0.9999982023429435}, {22.900000000000002, 0.9999982934814621}, {23.0, 0.9999983799860176}, {23.1, 0.9999984620930049}, 
 {23.200000000000003, 0.9999985400267126}, {23.3, 0.9999986139999483}, {23.400000000000002, 0.9999986842146262}, {23.5, 0.9999987508623289}, {23.6, 0.9999988141248376}, 
 {23.700000000000003, 0.9999988741746358}, {23.8, 0.9999989311753865}, {23.900000000000002, 0.9999989852823853}, {24.0, 0.9999990366429912}, {24.1, 0.9999990853970321}, 
 {24.200000000000003, 0.9999991316771913}, {24.3, 0.9999991756093762}, {24.400000000000002, 0.9999992173130628}, {24.5, 0.9999992569016275}, {24.6, 0.9999992944826599}, 
 {24.700000000000003, 0.9999993301582563}, {24.8, 0.9999993640253062}, {24.900000000000002, 0.9999993961757545}, {25.0, 1.0}};

        #region IManageable Members

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

        /// <summary>
        /// Member field on which the Configuration property relies.
        /// </summary>
        protected Configuration C;
        /// <summary>
        /// The configuration of the momentum estimator. Includes operational settings (such as momentum bounds) as well the specification of the material geometry.
        /// </summary>
        [XmlElement(typeof(MCSLikelihood.Configuration))]
        public SySal.Management.Configuration Config
        {
            get
            {
                return (SySal.Management.Configuration)(C.Clone());
            }
            set
            {
                C = (SySal.Processing.MCSLikelihood.Configuration)(value.Clone());
                C.Geometry = orderElementGeometry(C.Geometry);
                myChi = SearchConf(C.ConfidenceLevel);
            }
        }

        /// <summary>
        /// Allows the user to edit the supplied configuration.
        /// </summary>
        /// <param name="c">the configuration to be edited.</param>
        /// <returns><c>true</c> if the configuration has been modified, <c>false</c> otherwise.</returns>
        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            EditConfigForm ec = new EditConfigForm();
            ec.C = (SySal.Processing.MCSLikelihood.Configuration)(c.Clone());
            if (ec.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                c = (SySal.Processing.MCSLikelihood.Configuration)(ec.C.Clone());
                return true;
            }
            return false;
        }

        /// <summary>
        /// List of connections. It is always empty for MomentumEstimator.
        /// </summary>
        public IConnectionList Connections
        {
            get { return EmptyConnectionList; }
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

        #endregion
    }
}
