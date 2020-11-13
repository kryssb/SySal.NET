using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using System.IO;
using SySal.Processing;
using SySal.TotalScan;
using SySal.Processing.MCSAnnecy;

namespace SySal.Executables.BatchMCSAnnecy
{
    class Program
    {
        static void Main(string[] args)
        {
            bool inputFileDefined = false;
            bool inputConfigDefined = false;
            bool fitLoggerDefined = false;
            bool diffLoggerDefined = false;
            bool editconfigDefined = false;
            bool radlenDefined = false;
            bool minentrDefined = false;
            bool noise0Defined = false;
            bool noise1Defined = false;
            bool noise2Defined = false;
            bool zstepDefined = false;
            bool zmaxDefined = false;
            bool platedownDefined = false;
            bool useProjDefined = false;
            double uRL = -1.0;
            double radlen = -1.0;
            int minentr = -1;
            double noise0 = -1.0;
            double noise1 = -1.0;
            double noise2 = -1.0;
            string FileIn = "";
            string FileConfig = "";
            string FitLog = "";
            string DiffLog = "";
            double zstep = 1300.0;
            double zmax = 0.0;
            bool ignT = false;
            bool ignL = false;
            int platedown = 57;
            try
            {
                int argnum;
                for (argnum = 0; argnum < args.Length; argnum++)
                {
                    switch (args[argnum].ToLower())
                    {
                        case "/radlen":
                            {
                                radlenDefined = true;
                                radlen = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/minentr":
                            {
                                minentrDefined = true;
                                minentr = System.Convert.ToInt32(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/use":
                            {
                                useProjDefined = true;
                                switch (args[argnum + 1].ToLower())
                                {
                                    case "t": ignT = false; ignL = true; break;
                                    case "l": ignT = true; ignL = false; break;
                                    case "3d": ignT = false; ignL = false; break;
                                    default: throw new Exception("Unknown projection switch \"" + args[argnum + 1] + "\".");
                                }
                                argnum++;
                                break;
                            }
                        case "/noise0":
                            {
                                noise0Defined = true;
                                noise0 = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/noise1":
                            {
                                noise1Defined = true;
                                noise1 = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/noise2":
                            {
                                noise2Defined = true;
                                noise2 = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/zstep":
                            {
                                zstepDefined = true;
                                zstep = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/zmax":
                            {
                                zmaxDefined = true;
                                zmax = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/platedown":
                            {
                                platedownDefined = true;
                                platedown = System.Convert.ToInt32(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case "/fitlogfile":
                            {
                                fitLoggerDefined = true;
                                FitLog = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/difflogfile":
                            {
                                diffLoggerDefined = true;
                                DiffLog = args[argnum + 1];
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

                        case "/editconfigfile":
                            {
                                editconfigDefined = true;
                                FileConfig = args[argnum + 1];
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
                    System.Xml.Serialization.XmlSerializer xmls1 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MCSAnnecy.Configuration));
                    try
                    {
                        r1 = new System.IO.StreamReader(FileConfig);
                        ec.C = (SySal.Processing.MCSAnnecy.Configuration)xmls1.Deserialize(r1);
                    }
                    catch (Exception)
                    {
                        ec.C = (SySal.Processing.MCSAnnecy.Configuration)new SySal.Processing.MCSAnnecy.MomentumEstimator().Config;
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
                else if (inputFileDefined == false/* || inputConfigDefined == false*/) throw new Exception();

            }
            catch (Exception)
            {
                Console.WriteLine("Usage: BatchMCSAnnecy.exe {parameters}");
                Console.WriteLine("parameters");
                Console.WriteLine("/radlen <p> - radiation length (total) between two measurements");
                Console.WriteLine("/minentr <p> - minimum number of entries per cell");
                Console.WriteLine("/use <p> - can be T, L or 3D to use transverse, longitudinal or 3D data");
                Console.WriteLine("/noise0 <p> - 3D slope measurement error (0th order coefficient)");
                Console.WriteLine("/noise1 <p> - 3D slope measurement error (1st order coefficient)");
                Console.WriteLine("/noise2 <p> - 3D slope measurement error (2nd order coefficient)");
                Console.WriteLine("/zstep <p> - distance between two plates (default = 1300.0)");
                Console.WriteLine("/zmax <p> - Z of most downstream plate (default = 0.0)");
                Console.WriteLine("/platedown <p> - Id of the most downstream plate (default = 57)");
                Console.WriteLine("/inputfile <p> - input file");
                Console.WriteLine("/fitlogfile <p> - optional file to log the progress of fitting");
                Console.WriteLine("/difflogfile <p> - optional file to log slope differences");
                Console.WriteLine("/configfile<p> - geometry configuration file");
                Console.WriteLine("/editconfigfile <p> - geometry configuration file to be edited/created");
                Console.WriteLine();
                Console.WriteLine("Input data format:");
                Console.WriteLine("ID\tPlate\tZ\tSX\tSY\tX\tY");
                return;
            }

            SySal.Processing.MCSAnnecy.Configuration C = new SySal.Processing.MCSAnnecy.Configuration();
            if (noise0Defined) C.SlopeError3D_0 = noise0;
            if (noise1Defined) C.SlopeError3D_1 = noise1;
            if (noise2Defined) C.SlopeError3D_2 = noise2;
            if (minentrDefined) C.MinEntries = minentr;
            if (radlenDefined) C.RadiationLength = radlen;
            C.IgnoreLongitudinal = ignL;
            C.IgnoreTransverse = ignT;

            if (FileConfig != null && FileConfig.Trim().Length > 0)
            {
                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Geometry));
                System.IO.StreamReader r = new System.IO.StreamReader(FileConfig);
                r.Close();
            }

            System.Text.RegularExpressions.Regex colfmt_rx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");
            SySal.Processing.MCSAnnecy.MomentumEstimator momCalculation = new SySal.Processing.MCSAnnecy.MomentumEstimator();
            momCalculation.Config = C;

            System.IO.StreamWriter logstream = null;
            if (FitLog.Trim().Length > 0)
            {
                logstream = new System.IO.StreamWriter(FitLog);
                logstream.AutoFlush = true;
                momCalculation.FitLog = logstream;
            }

            System.IO.StreamWriter difflogstream = null;
            if (DiffLog.Trim().Length > 0)
            {
                difflogstream = new System.IO.StreamWriter(DiffLog);
                difflogstream.AutoFlush = true;
                momCalculation.DiffLog = difflogstream;
            }

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
                    info.Field = (uint)System.Convert.ToDouble(rx_m.Groups[2].Value);
                    //info.Field = (uint)(Math.Round((info.Intercept.Z - zmax) / zstep) + platedown);
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
            if (logstream != null) logstream.Close();
            if (difflogstream != null) difflogstream.Close();
        }





        private static void ComputeAndShowResult(string FileIn, int idTrack0, System.Collections.ArrayList InputData, SySal.Processing.MCSAnnecy.MomentumEstimator mcs)
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

