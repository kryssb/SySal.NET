using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Imaging
{
    [Serializable]
    public struct Cluster3D
    {
        public uint Layer;
        public double Z;
        public Cluster Cluster;
    }

    [Serializable]
    public class Grain3D
    {
        public SySal.Imaging.Cluster3D[] Clusters;
        public uint Volume;
        public SySal.BasicTypes.Vector Position;
        public SySal.BasicTypes.Vector Tangent;
        public uint FirstLayer;
        public uint LastLayer;
        public SySal.BasicTypes.Vector TopExtent;
        public SySal.BasicTypes.Vector BottomExtent;

        static public Grain3D FromClusterCenters(SySal.Imaging.Cluster3D[] cls)
        {
            Grain3D g = new Grain3D();
            g.Clusters = cls;
            g.Volume = 0;
            g.FirstLayer = g.LastLayer = cls[0].Layer;
            foreach (SySal.Imaging.Cluster3D c in cls)
            {
                g.Volume += c.Cluster.Area;
                if (g.FirstLayer > c.Layer) g.FirstLayer = c.Layer;
                else if (g.LastLayer < c.Layer) g.LastLayer = c.Layer;
                g.Position.X += c.Cluster.X * c.Cluster.Area;
                g.Position.Y += c.Cluster.Y * c.Cluster.Area;
                g.Position.Z += c.Z * c.Cluster.Area;
            }
            g.Position.X /= g.Volume;
            g.Position.Y /= g.Volume;
            g.Position.Z /= g.Volume;
            return g;
        }
    }

    /// <summary>
    /// Classes that make grains from clusters implement this interface.
    /// </summary>
    public interface IGrain3DMaker
    {
        /// <summary>
        /// Makes a list of Grains3D from clusters with 3D information.
        /// </summary>
        /// <param name="cls">the layers of clusters</param>
        /// <param name="Positions">the positions of the centers of the field of view.</param>
        /// <returns>the list of Grains3D.</returns>
        Grain3D[] MakeGrainsFromClusters(Cluster3D[][] cls, SySal.BasicTypes.Vector [] positions);
    }
}
