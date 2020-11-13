using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using System.IO;

namespace SySal.Executables.TagPrimary
{
    class program
    {
        static void Main(string[] args)
        {
            bool inputFileDefined  = false;
            bool outputFileDefined = false;
            bool eventIdDefined = false;
            bool xmlConfigDefined = false;
            string fileIn  = "";
            string fileOut = "";
            string xmlConfig = "";
            long eventID = 0L;
            int argnum;

            try
            {
                for (argnum = 0; argnum < args.Length; argnum++)
                {
                    switch (args[argnum].ToLower())
                    {
                        case "/inputfile":
                            {
                                inputFileDefined = true;
                                fileIn = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/outputfile":
                            {
                                outputFileDefined = true;
                                fileOut = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case "/eventid":
                            {
                                eventIdDefined = true;
                                eventID = SySal.OperaDb.Convert.ToInt64(args[argnum+1]);
                                argnum++;
                                break;
                            }
                        case "/xmlconfig":
                            {
                                xmlConfigDefined = true;
                                xmlConfig = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        default: throw new Exception("Unknown projection switch \"" + args[argnum + 1] + "\".");
                    }
                }
                if (!inputFileDefined || !outputFileDefined) throw new Exception();

                if (!eventIdDefined) 
                {
                    eventID = 1;
                }

                SySal.Processing.TagPrimary.Configuration C = new SySal.Processing.TagPrimary.Configuration();

                // Create Configuration Object For TagPrimary
                // new SySal.Processing.TagPrimary.Configuration("Configuration For TagPrimary");
                if (xmlConfigDefined)
                {
                    string xml = ((SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(xmlConfig, typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings))).Settings;
                    System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.TagPrimary.Configuration));
                    C = (SySal.Processing.TagPrimary.Configuration)xmls.Deserialize(new System.IO.StringReader(xml));
                }
               
                
                // Create an istance of the object TagPrimary.
                SySal.Processing.TagPrimary.PrimaryVertexTagger tagPrim = new SySal.Processing.TagPrimary.PrimaryVertexTagger();
                //configure it!
                tagPrim.Config = C;
                // Read TSR file (with datasets TSR,CS,SBSF inside)
                tagPrim.ReadTSR(fileIn);

                // Set the eventID
                tagPrim.setEventID(eventID);
                
                // Start primary search
                SySal.Processing.TagPrimary.PrimaryVertexTagger.TagPrimaryResult outputInfo = tagPrim.SearchPrimaryVertex();

                //bool isFound = outputInfo.isFound; ;
                Console.WriteLine("\nEvent: " + outputInfo.EventId.ToString() +
                    "\nType: " + outputInfo.EventType +
                    "\nScanBackPaths: " + outputInfo.ScanBackPaths.ToString() +
                    "\nScanBackTracks: " + outputInfo.ScanBackPathsInVolume.ToString() +
                    "\nCSpaths: " + outputInfo.CSPaths.ToString() +
                    "\nCStracks: " + outputInfo.CSPathsInVolume.ToString() +
                    "\nPrimaryFound: " + outputInfo.IsFound.ToString());
                if (outputInfo.IsFound)
                {
                    Console.WriteLine(
                    "\nVertexID: " + outputInfo.VertexId.ToString() +
                    "\nnProng: " + outputInfo.Prongs.ToString());
                }
                
                // Write TSR with flagged vertex, flagged muon....(if any)
                tagPrim.WriteTSR(fileOut);  
            }
            catch (Exception x)
            {
                Console.WriteLine("Exception: " + x.Message + "\r\nSource:\r\n" + x.ToString());
                Console.WriteLine("Usage: TagPrimary.exe {parameters}");
                Console.WriteLine("parameters");
                Console.WriteLine("/inputfile  <p> - .tsr input file");
                Console.WriteLine("/outputfile <p> - .tsr output file with vertex and tracks tagged");
                Console.WriteLine("/eventid    <p> - event id number");
                Console.WriteLine("/xmlconfig  <p> - xml configuration");

                return;
            }
        }
    }
}
