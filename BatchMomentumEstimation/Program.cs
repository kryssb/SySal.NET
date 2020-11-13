using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using System.IO;
using SySal.Processing.MCSLikelihood;
using SySal.TotalScan;

namespace SySal.Executables.BatchMomentumEstimation
{
    class Program
    {
        static void Main(string[] args)
        {
            bool minpDefined = false;
            bool maxpDefined = false;
            bool stepPDefined = false;
            bool inputFileDefined = false;
            bool inputConfigDefined = false;
            bool angDiffDumpDefined = false;
            bool tkDumpDefined = false;
            bool cvDumpDefined = false;
            bool lkDumpDefined = false;
            bool lvCfDefined = false;
            bool editconfigDefined = false;
            bool radlenDefined = false;
            bool noiseDefined = false;
            double pMin = -1.0;
            double pMax = -1.0;
            double stepP = -1.0;
            double lvCf = -1.0;
            double uRL = -1.0;
            double noise = -1.0;
            string FileIn = "";
            string FileConfig = "";
            string tkDumpFile = null;
            string cvDumpFile = null;
            string lkDumpFile = null;
            string angDiffDumpFile = null;
            try
            {
                int argnum;
                for (argnum = 0; argnum < args.Length; argnum++)
                {
                    switch (args[argnum].ToLower())
                    {
                        case "/noise":
                            {
                                noiseDefined = true;
                                noise = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/minp":
                            {
                                minpDefined = true;
                                pMin = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/maxp":
                            {
                                maxpDefined = true;
                                pMax = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/stepp":
                            {
                                stepPDefined = true;
                                stepP = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/inputfile":
                            {
                                inputFileDefined = true;
                                FileIn = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/configfile":
                            {
                                inputConfigDefined = true;
                                FileConfig = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/cl":
                            {
                                lvCfDefined = true;
                                lvCf = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }

                        case "/angdiffdump":
                            {
                                angDiffDumpDefined = true;
                                angDiffDumpFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/trackingdump":
                            {
                                tkDumpDefined = true;
                                tkDumpFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/covariancedump":
                            {
                                cvDumpDefined = true;
                                cvDumpFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/likelihooddump":
                            {
                                lkDumpDefined = true;
                                lkDumpFile = args[argnum + 1];
                                argnum++;
                                break;
                            }

                        case "/editconfigfile":
                            {
                                editconfigDefined = true;
                                FileConfig = args[argnum + 1];
                                break;
                            }
                        case "/radlen":
                            {
                                radlenDefined = true;
                                uRL = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                break;
                            }
                    }

                }

                if (editconfigDefined)
                {
                    System.Windows.Forms.Application.EnableVisualStyles();
                    EditConfigForm ec = new EditConfigForm();
                    System.IO.StreamReader r1 = null;
                    System.IO.StreamWriter w = null;
                    System.Xml.Serialization.XmlSerializer xmls1 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MCSLikelihood.Configuration));
                    try
                    {
                        r1 = new System.IO.StreamReader(FileConfig);
                        ec.C = (SySal.Processing.MCSLikelihood.Configuration)xmls1.Deserialize(r1);
                    }
                    catch (Exception)
                    {
                        ec.C = (SySal.Processing.MCSLikelihood.Configuration)new SySal.Processing.MCSLikelihood.MomentumEstimator().Config;
                    }
                    finally
                    {
                        if (r1 != null) r1.Close();
                    }
                    if (ec.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                    {
                        try
                        {
                            w = new StreamWriter(FileConfig);
                            xmls1.Serialize(w, ec.C);
                            w.Flush();
                            w.Close();
                        }
                        catch (Exception x)
                        {
                            System.Windows.Forms.MessageBox.Show("Can't write to file \"" + FileConfig + "\"!", "File Error", System.Windows.Forms.MessageBoxButtons.OK, System.Windows.Forms.MessageBoxIcon.Error);
                        }
                        finally
                        {
                            if (w != null) w.Close();
                        }
                    }
                    return;
                }
                else if (noiseDefined == false || minpDefined == false || maxpDefined == false || stepPDefined == false || inputFileDefined == false || inputConfigDefined == false || lvCfDefined == false || radlenDefined == false) throw new Exception();

            }
            catch (Exception)
            {
                Console.WriteLine("Usage: BatchMomentumEstimation.exe {parameters}");
                Console.WriteLine("parameters");
                Console.WriteLine("/noise <p> - measurement error");
                Console.WriteLine("/minp <p> - minimum momentum");
                Console.WriteLine("/maxp <p> - maximum momentum");
                Console.WriteLine("/stepp <p> - step momentum");
                Console.WriteLine("/cl <p> - confidence level");
                Console.WriteLine("/inputfile <p> - input file");
                Console.WriteLine("/configfile<p> - geometry configuration file");
                Console.WriteLine("/angdiffdump <p> - angular difference dump to debug");
                Console.WriteLine("/trackingdump <p> - tracking dump to debug");
                Console.WriteLine("/covariancedump <p> - covariance dump to debug");
                Console.WriteLine("/likelihooddump <p> - likelihood dump to debug");
                Console.WriteLine("/editconfigfile <p> - geometry configuration file to be edited/created");
                Console.WriteLine("/radlen <p> - unit of Radiation Length");
                Console.WriteLine();
                Console.WriteLine("Input data format:");
                Console.WriteLine("ID\tPlate\tZ\tSX\tSY\tX\tY");
                return;
            }

            SySal.Processing.MCSLikelihood.Configuration C = new SySal.Processing.MCSLikelihood.Configuration();
            C.ConfidenceLevel = lvCf;
            C.MaximumMomentum = pMax;
            C.MinimumMomentum = pMin;
            C.MomentumStep = stepP;
            C.SlopeError = noise;
            C.MinimumRadiationLengths = uRL; // da parametri!            
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Geometry));
            System.IO.StreamReader r = new System.IO.StreamReader(FileConfig);
            C.Geometry = (Geometry)xmls.Deserialize(r);
            r.Close();

            System.Text.RegularExpressions.Regex colfmt_rx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");
            SySal.Processing.MCSLikelihood.MomentumEstimator momCalculation = new SySal.Processing.MCSLikelihood.MomentumEstimator();
            momCalculation.Config = C;
            if (angDiffDumpDefined == true) { momCalculation.AngularDiffDumpFile = angDiffDumpFile; }
            if (tkDumpDefined == true) { momCalculation.TrackingDumpFile = tkDumpFile; }
            if (cvDumpDefined == true) { momCalculation.CovarianceDumpFile = cvDumpFile; }
            if (lkDumpDefined == true) { momCalculation.LikelihoodDumpFile = lkDumpFile; }

            //legge il file con i dati


            ArrayList InputData = new ArrayList();

            int idTrack;
            int idTrack0 = -1;
            bool isfirst = true;
            StreamReader FileInput = new StreamReader(FileIn);
            string rLine;

            while ((rLine = FileInput.ReadLine()) != null)
            {
                System.Text.RegularExpressions.Match rx_m = colfmt_rx.Match((string)(rLine));
                if (rx_m.Success)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    idTrack = System.Convert.ToInt32(rx_m.Groups[1].Value);
                    info.Slope.X = System.Convert.ToDouble(rx_m.Groups[4].Value);
                    info.Slope.Y = System.Convert.ToDouble(rx_m.Groups[5].Value);
                    info.Intercept.Z = System.Convert.ToDouble(rx_m.Groups[3].Value);
                    info.Intercept.X = System.Convert.ToDouble(rx_m.Groups[6].Value);
                    info.Intercept.Y = System.Convert.ToDouble(rx_m.Groups[7].Value);
                    if (isfirst) idTrack0 = idTrack;
                    if (idTrack0 != idTrack)
                    {
                        ComputeAndShowResult(FileIn, idTrack0, InputData, momCalculation);
                        InputData.Clear();
                        idTrack0 = idTrack;
                    }
                    isfirst = false;
                    InputData.Add(info);
                }
                else
                {
                    Console.WriteLine("Format error at line " + rLine);
                    return;
                }


            }
            if (InputData.Count > 0) ComputeAndShowResult(FileIn, idTrack0, InputData, momCalculation);
            FileInput.Close();
        }





        private static void ComputeAndShowResult(string FileIn, int idTrack0, System.Collections.ArrayList InputData, SySal.Processing.MCSLikelihood.MomentumEstimator mcs)
        {
            MomentumResult finalResult = new MomentumResult();
#if !(DEBUG)
            try
            {
#endif
                finalResult = mcs.ProcessData((SySal.Tracking.MIPEmulsionTrackInfo[])InputData.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo)));
#if !(DEBUG)
            }
            catch (Exception x)
            {
                finalResult.Value = finalResult.ConfidenceLevel = finalResult.LowerBound = finalResult.UpperBound = 0.0;
                Console.WriteLine(x.ToString());
            }
#endif
            Console.WriteLine(FileIn + " " + idTrack0 + " " + finalResult.Value + " " + finalResult.LowerBound + " " + finalResult.UpperBound);
        }
    }

}
  
