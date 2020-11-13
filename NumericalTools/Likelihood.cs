using System;

namespace NumericalTools
{
    /// <summary>
    /// The likelihood function for track fitting results.
    /// </summary>
    public interface Likelihood
    {
        /// <summary>
        /// The number of parameters for this likelihood.
        /// </summary>
        int Parameters { get; }
        /// <summary>
        /// Retrieves the name of a parameter.
        /// </summary>
        /// <param name="iparam">the number of the parameter whose name is sought.</param>
        /// <returns>the name of the parameter.</returns>
        string ParameterName(int iparam);
        /// <summary>
        /// Retrieves the minimum acceptable value for a parameter.
        /// </summary>
        /// <param name="iparam">the number of the parameter whose minimum bound is sought.</param>
        /// <returns>the value of the minimum bound.</returns>
        double MinBound(int iparam);
        /// <summary>
        /// Retrieves the maximum acceptable value for a parameter.
        /// </summary>
        /// <param name="iparam">the number of the parameter whose maximum bound is sought.</param>
        /// <returns>the value of the maximum bound.</returns>
        double MaxBound(int iparam);
        /// <summary>
        /// Computes the value of the likelihood function for a given set of parameters.
        /// </summary>
        /// <param name="paramvalues">the parameter values.</param>
        /// <returns>the value of the likelihood function.</returns>
        double Value(params double[] paramvalues);
        /// <summary>
        /// Computes the natural logarithm of the value of the likelihood function for a given set of parameters.
        /// </summary>
        /// <param name="paramvalues">the parameter values.</param>
        /// <returns>the natural logarithm of the value of the likelihood function.</returns>
        double LogValue(params double[] paramvalues);
        /// <summary>
        /// Yields the most likely value for a parameter.
        /// </summary>
        double Best(int iparam);
        /// <summary>
        /// Computes the confidence regions for one parameter, marginalizing all others.
        /// </summary>
        /// <param name="cl">the confidence level to use to compute the region.</param>
        /// <returns>the extents of the intervals of the confidence region.</returns>
        /// <remarks>The number of elements in the returned array is always even: each even element is the lower 
        /// bound of an interval and each next odd element is the upper bound of the same interval as shown in the example.
        /// <example><c>1,2.5,5,8.8</c></example>
        /// In this case the first interval spans [1,2.5], and the second spans [5,8.8].</remarks>
        double[] ConfidenceRegions(int i, double cl);
    }

    /// <summary>
    /// One-parameter likelihood.
    /// </summary>
    public class OneParamLogLikelihood : Likelihood
    {
        protected double[] m_logvalues;

        protected double m_pMin;

        protected double m_pMax;

        protected int m_IBest;

        protected string m_ParamName;

        protected double m_pStep;
        /// <summary>
        /// Builds a one-parameter likelihood defined using its logarithms on a regular grid of parameter values.
        /// </summary>
        /// <param name="pmin">the minimum value of the parameter.</param>
        /// <param name="pmax">the maximum value of the parameter.</param>
        /// <param name="logvalues">array of values of the logarithm of the likelihood function.</param>
        public OneParamLogLikelihood(double pmin, double pmax, double[] logvalues, string paramname)
        {
            m_pMin = pmin;
            m_pMax = pmax;
            m_pStep = (logvalues.Length > 1) ? ((pmax - pmin) / (logvalues.Length - 1)) : 1.0;
            m_logvalues = (double[])logvalues.Clone();
            m_IBest = 0;
            int i;
            for (i = 1; i < m_logvalues.Length; i++)
                if (m_logvalues[i] > m_logvalues[m_IBest])
                    m_IBest = i;
        }

        #region LikelihoodFunction Members
        /// <summary>
        /// Returns 1.
        /// </summary>
        public int Parameters
        {
            get { return 1; }
        }
        /// <summary>
        /// The name of the parameter.
        /// </summary>
        /// <param name="iparam">must be 0.</param>
        /// <returns>the name of the parameter.</returns>
        public string ParameterName(int iparam)
        {
            if (iparam != 0) throw new Exception("This function has one parameter.");
            return (string)m_ParamName.Clone();
        }
        /// <summary>
        /// The minimum value for the parameter.
        /// </summary>
        /// <param name="iparam">must be 0.</param>
        /// <returns>the minimum value of the parameter.</returns>
        public double MinBound(int iparam)
        {
            if (iparam != 0) throw new Exception("This function has one parameter.");
            return m_pMin;
        }
        /// <summary>
        /// The maximum value for the parameter.
        /// </summary>
        /// <param name="iparam">must be 0.</param>
        /// <returns>the maximum value of the parameter.</returns>
        public double MaxBound(int iparam)
        {
            if (iparam != 0) throw new Exception("This function has one parameter.");
            return m_pMax;
        }
        /// <summary>
        /// Computes the value of the likelihood function.
        /// </summary>
        /// <param name="paramvalues">the value of the parameter.</param>
        /// <returns>the value of the likelihood function.</returns>
        /// <remarks>the value of the function is computed by interpolating its logarithm.</remarks>
        public double Value(params double[] paramvalues)
        {
            return Math.Exp(LogValue(paramvalues));
        }
        /// <summary>
        /// Computes the value of the logarithm of the likelihood function.
        /// </summary>
        /// <param name="paramvalues">the value of the parameter.</param>
        /// <returns>the value of the logarithm of the likelihood function, computed as a first order spline.</returns>
        public double LogValue(params double[] paramvalues)
        {
            if (paramvalues.Length != 1) throw new Exception("The method or operation is not implemented.");
            double p = paramvalues[0];
            if (p < m_pMin || p > m_pMax) throw new Exception("Parameter out of bounds");
            if (m_pMin == m_pMax) return m_logvalues[0];
            int ip = (int)((p - m_pMin) / m_pStep);
            double plambda = (p - ip * m_pStep - m_pMin) / m_pStep;
            if (Math.Abs(plambda) < 1e-4) return m_logvalues[ip];
            return m_logvalues[ip] * (1.0 - plambda) + m_logvalues[ip + 1] * plambda;
        }

        /// <summary>
        /// Computes the confidence regions for the parameter.
        /// </summary>
        /// <param name="cl">the confidence level to use to compute the region.</param>
        /// <param name="i">the parameter whose confidence region is needed.</param>
        /// <returns>the extents of the intervals of the confidence region.</returns>
        /// <remarks>The number of elements in the returned array is always even: each even element is the lower 
        /// bound of an interval and each next odd element is the upper bound of the same interval as shown in the example.
        /// <example><c>1,2.5,5,8.8</c></example>
        /// In this case the first interval spans [1,2.5], and the second spans [5,8.8].</remarks>
        public double[] ConfidenceRegions(int i, double cl)
        {
            double maxlkl = m_logvalues[m_IBest];
            double cutlkl = maxlkl - 0.5 * SearchConf(cl);
            System.Collections.ArrayList m_extents = new System.Collections.ArrayList();
            bool isregion = false;
            for (i = 0; i < m_logvalues.Length; i++)
                if (isregion)
                {
                    if (m_logvalues[i] < cutlkl)
                    {
                        m_extents.Add(i * m_pStep + m_pMin);
                        isregion = false;
                    }
                }
                else
                {
                    if (m_logvalues[i] >= cutlkl)
                    {
                        m_extents.Add(i * m_pStep + m_pMin);
                        isregion = true;
                    }
                }
            if (isregion) m_extents.Add(m_pMax);
            return (double[])(m_extents.ToArray(typeof(double)));
        }

        /// <summary>
        /// Yields the most likely value for the parameter.
        /// </summary>
        public double Best(int iparam)
        {
            if (iparam != 0) throw new Exception("This function has one parameter.");
            return m_IBest * m_pStep + m_pMin;
        }

        #endregion


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

            result = Chi2Interpolation(lowLimit, upLimit, lvlConf);

            return result;
        }

        private double Chi2Interpolation(int lw, int up, double lConf)
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

        private static double[,] Chi2Integ = new double[,] 
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


        /// <summary>
        /// Builds a one-parameter likelihood obtained as a product of several independent one-parameter likelihoods with the same parameter.
        /// </summary>
        /// <param name="samplingstep">the step to be used to resample the final likelihood.</param>
        /// <param name="multipliers">the likelihood functions to be multiplied.</param>
        public OneParamLogLikelihood(double samplingstep, params Likelihood [] multipliers)
        {
            m_pStep = samplingstep;
            m_pMin = multipliers[0].MinBound(0);
            m_pMax = multipliers[0].MaxBound(0);
            int i;
            for (i = 1; i < multipliers.Length; i++)
            {
                if (m_pMin > multipliers[i].MinBound(0)) m_pMin = multipliers[i].MinBound(0);
                if (m_pMax > multipliers[i].MaxBound(0)) m_pMax = multipliers[i].MaxBound(0);
            }
            if (m_pMin > m_pMax) throw new Exception("Null allowed region.");
            int nvalues;
            if (m_pMin == m_pMax)
            {
                m_pStep = 1.0;
                nvalues = 1;
            }
            else
            {
                nvalues = (int)Math.Floor((m_pMax - m_pMin) / m_pStep) + 1;
                m_pStep = (m_pMax - m_pMin) / (nvalues - 1);
            }
            m_logvalues = new double[nvalues];
            int ip;
            for (ip = 0; ip < nvalues; ip++)
                for (i = 0; i < multipliers.Length; i++)
                    m_logvalues[ip] += multipliers[i].LogValue(ip * m_pStep + m_pMin);
            m_IBest = 0;            
            for (i = 1; i < m_logvalues.Length; i++)
                if (m_logvalues[i] > m_logvalues[m_IBest])
                    m_IBest = i;
        }
    }
}