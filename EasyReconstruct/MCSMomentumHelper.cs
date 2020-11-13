using System;
using System.Collections.Generic;
using System.Text;
using SySal;
using SySal.Management;
using SySal.TotalScan;

namespace SySal.Executables.EasyReconstruct
{
    internal class MCSMomentumHelper
    {
        public static MomentumResult ProcessData(IMCSMomentumEstimator algo, Track tk)
        {
            int i, j;
            SySal.Tracking.MIPEmulsionTrackInfo[] tkinfo = new SySal.Tracking.MIPEmulsionTrackInfo[tk.Length];
            if (algo is SySal.Processing.MCSLikelihood.MomentumEstimator)
            {
                Geometry geom = ((SySal.Processing.MCSLikelihood.Configuration)(((SySal.Management.IManageable)((object)algo)).Config)).Geometry;                
                for (i = 0; i < tkinfo.Length; i++)
                {
                    if (tk[i].LayerOwner.Side != 0) throw new Exception("This functionality is only available for base-tracks.\r\nConsider using the interactive version, available in the interactive display.");
                    tkinfo[i] = tk[i].Info;
                    for (j = 0; j < geom.Layers.Length && tk[i].LayerOwner.SheetId != geom.Layers[j].Plate; j++) ;
                    if (j == geom.Layers.Length) throw new Exception("Layer " + i + " Sheet " + tk[i].LayerOwner.SheetId + " does not map to any plate in momentum geometry.\r\nPlease check your geometry information.");
                    double z = (tkinfo[i].TopZ + tkinfo[i].BottomZ) * 0.5;
                    if (z < tk[i].LayerOwner.UpstreamZ) z = tk[i].LayerOwner.UpstreamZ;
                    else if (z > tk[i].LayerOwner.DownstreamZ) z = tk[i].LayerOwner.DownstreamZ;
                    tkinfo[i].Intercept.X += tkinfo[i].Slope.X * (z - tkinfo[i].Intercept.Z);
                    tkinfo[i].Intercept.Y += tkinfo[i].Slope.Y * (z - tkinfo[i].Intercept.Z);
                    tkinfo[i].Intercept.Z = geom.Layers[j].ZMin + z - tk[i].LayerOwner.UpstreamZ;
                }                
            }
            else if (algo is SySal.Processing.MCSAnnecy.MomentumEstimator)
            {
                for (i = 0; i < tkinfo.Length; i++)
                {
                    if (tk[i].LayerOwner.Side != 0) throw new Exception("This functionality is only available for base-tracks.\r\nConsider using the interactive version, available in the interactive display.");
                    tkinfo[i] = tk[i].Info;
                    tkinfo[i].Field = (uint)tk[i].LayerOwner.Id;                    
                }
            }
            else throw new Exception("Unsupported algorithm.");
            MomentumResult res = algo.ProcessData(tkinfo);
            if (res.Value < 0.0) throw new Exception("Invalid measurement");
            tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"), res.Value);
            tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMin" + (res.ConfidenceLevel * 100.0).ToString("F0", System.Globalization.CultureInfo.InvariantCulture)), res.LowerBound);
            tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMax" + (res.ConfidenceLevel * 100.0).ToString("F0", System.Globalization.CultureInfo.InvariantCulture)), res.UpperBound);
            return res;
        }
    }
}
