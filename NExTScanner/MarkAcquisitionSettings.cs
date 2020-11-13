using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    /// <summary>
    /// Settings for mark acquisition.
    /// </summary>
    [Serializable]
    public class MarkAcquisitionSettings : SySal.Management.Configuration, ICloneable
    {
        /// <summary>
        /// XY speed to approach marks.
        /// </summary>        
        public double XYSpeed;

        /// <summary>
        /// XY acceleration to approach marks.
        /// </summary>        
        public double XYAcceleration;

        /// <summary>
        /// XY speed tolerance to complete approach motion to marks.
        /// </summary>        
        public double XYPosTolerance;

        /// <summary>
        /// Builds a new set of mark acquisition parameters.
        /// </summary>
        public MarkAcquisitionSettings()
            : base("")
        {
            XYSpeed = 10000.0;
            XYAcceleration = 20000.0;
            XYPosTolerance = 50.0;
        }

        /// <summary>
        /// Builds a new set of mark acquisition parameters with specified name.
        /// </summary>
        /// <param name="name">the name of the configuration.</param>
        public MarkAcquisitionSettings(string name)
            : base(name)
        {
            XYSpeed = 10000.0;
            XYAcceleration = 20000.0;
            XYPosTolerance = 50.0;
        }

        public override object Clone()
        {
            MarkAcquisitionSettings m = new MarkAcquisitionSettings();

            m.XYSpeed = XYSpeed;
            m.XYPosTolerance = XYPosTolerance;
            m.XYAcceleration = XYAcceleration;

            return m;
        }

        internal static MarkAcquisitionSettings Default
        {
            get
            {
                MarkAcquisitionSettings c = SySal.Management.MachineSettings.GetSettings(typeof(MarkAcquisitionSettings)) as MarkAcquisitionSettings;
                if (c == null) c = new MarkAcquisitionSettings();
                return c;
            }
        }


        public void Copy(MarkAcquisitionSettings c)
        {
            XYPosTolerance = c.XYPosTolerance;
            XYAcceleration = c.XYAcceleration;
            XYSpeed = c.XYSpeed;
        }

        public const string FileExtension = "markacq.config";

    }
}