using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.CSRelatedGraphSelector
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 7)
            {
                Console.WriteLine("usage: CSRelatedGraphSelector.exe <tlg file> <ascii file> <brick> <plate> <mingrains> <postol> <slopetol>");
                Console.WriteLine("output: BRICK PLATE GRAINS PX PY PZ SX SY SIGMA DPX DPY DSX DSY");
                return;
            }
            int brickid = Convert.ToInt32(args[2]);
            if (brickid > 1999999)
            {
                Console.WriteLine("No track to select on a CS.");
                System.IO.File.WriteAllText(args[1], "\r\n");
                return;
            }
            SySal.OperaDb.OperaDbConnection conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
            conn.Open();
            int csid = brickid;
            try
            {
                csid = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT MAX(ID) FROM TB_EVENTBRICKS WHERE MOD(ID,1000000) = " + (brickid - (brickid / 1000000) * 1000000), conn).ExecuteScalar());
            }
            catch (Exception) { }
            SySal.OperaDb.OperaDbDataReader csr = new SySal.OperaDb.OperaDbCommand("SELECT (ID_PLATE - 1) * 300 + 4850 as Z, POSX, POSY, SLOPEX, SLOPEY FROM VW_LOCAL_CS_CANDIDATES WHERE ID_CS_EVENTBRICK = " + csid, conn).ExecuteReader();
            System.Collections.ArrayList arr = new System.Collections.ArrayList();
            while (csr.Read())
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                info.Intercept.Z = csr.GetDouble(0);
                info.Intercept.X = csr.GetDouble(1);
                info.Intercept.Y = csr.GetDouble(2);
                info.Slope.X = csr.GetDouble(3);
                info.Slope.Y = csr.GetDouble(4);
                arr.Add(info);
            }
            csr.Close();
            double Z = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + brickid + " AND ID = " + Convert.ToInt32(args[3]).ToString(), conn).ExecuteScalar());
            conn.Close();
            SySal.DataStreams.OPERALinkedZone lz = new SySal.DataStreams.OPERALinkedZone(args[0]);
            int basen = lz.Length;
            int i;
            int mingrains = Convert.ToInt32(args[4]);
            double postol = Convert.ToDouble(args[5]);
            double slopetol = Convert.ToDouble(args[6]);            
            string outstr = "";
            int sel = 0;
            for (i = 0; i < basen; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo lzinfo = lz[i].Info;
                if (lzinfo.Count < mingrains) continue;
                foreach (SySal.Tracking.MIPEmulsionTrackInfo csinfo in arr)
                {
                    double dz = Math.Abs(csinfo.Intercept.Z - Z);
                    double slopescatt = 0.014 * Math.Sqrt(dz / (1.3 * 5600.0)); /* assume 1 GeV */
                    double posscatt = slopescatt * dz / Math.Sqrt(3.0);                    
                    double dsx = csinfo.Slope.X - lzinfo.Slope.X;
                    double dsy = csinfo.Slope.Y - lzinfo.Slope.Y;
                    if (dsx * dsx + dsy * dsy > (2.0 * (slopescatt * slopescatt + slopetol * slopetol))) continue;
                    double dpx = csinfo.Intercept.X + (Z - csinfo.Intercept.Z) * (csinfo.Slope.X + lzinfo.Slope.X) * 0.5 - lzinfo.Intercept.X;
                    double dpy = csinfo.Intercept.Y + (Z - csinfo.Intercept.Z) * (csinfo.Slope.Y + lzinfo.Slope.Y) * 0.5 - lzinfo.Intercept.Y;
                    if (dpx * dpx + dpy * dpy > (2.0 * (posscatt * posscatt + postol * postol))) continue;
                    outstr += "\r\n" + brickid + " " + args[3] + " " + lzinfo.Count + " " + lzinfo.Intercept.X + " " + lzinfo.Intercept.Y + " " + Z + " " + lzinfo.Slope.X + " " + lzinfo.Slope.Y + " " + lzinfo.Sigma + " " + dpx + " " + dpy + " " + dsx + " " + dsy;
                    sel++;
                    break;
                }
            }
            Console.WriteLine("Selected: " + sel + " track(s)");
            System.IO.File.WriteAllText(args[1], outstr);
        }
    }
}
