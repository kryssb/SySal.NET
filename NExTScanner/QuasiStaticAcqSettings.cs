using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    /// <summary>
    /// Settings for quasi-static acquisition.
    /// </summary>
    [Serializable]
    public class QuasiStaticAcqSettings : SySal.Management.Configuration, ICloneable
    {
        /// <summary>
        /// The axis to move during scanning.
        /// </summary>
        [Serializable]
        public enum MoveAxisForScan { X, Y };
        /// <summary>
        /// Defines the sides to be scanned.
        /// </summary>
        [Serializable]
        public enum ScanMode { Both, Top, Bottom };
        /// <summary>
        /// The number of layers where images must be acquired.
        /// </summary>
        public uint Layers = 21;
        /// <summary>
        /// The pitch between layers.
        /// </summary>
        public double Pitch = 3.0;
        /// <summary>
        /// Total Z sweep including all layers.
        /// </summary>
        public double ZSweep { get { return (Layers - 1) * Pitch; } }
        /// <summary>
        /// Layers exceeding this threshold are considered to be in the emulsion layer.
        /// </summary>
        public uint ClusterThreshold = 3000;
        /// <summary>
        /// A view is valid if at the number of layers with at least <c>ClusterThreshold</c> clusters exceeds this number.
        /// </summary>
        public uint MinValidLayers = 12;
        /// <summary>
        /// The emulsion is searched for using this tolerance on Z.
        /// </summary>
        public double FocusSweep = 70.0;
        /// <summary>
        /// The expected thickness for the plastic base of the emulsion, if any.
        /// </summary>
        public double BaseThickness = 205.0;
        /// <summary>
        /// The expected thickness for emulsion layers.
        /// </summary>
        public double EmulsionThickness = 45.0;
        /// <summary>
        /// The side(s) to be scanned.
        /// </summary>
        public ScanMode Sides = ScanMode.Both;
        /// <summary>
        /// Overlap in micron between adjacent views.
        /// </summary>
        public double ViewOverlap = 30.0;
        /// <summary>
        /// Defines the fraction of field of view motion executed during the Z sweep. Set to 0 to disable emulation of continuous motion (layer skewing in X/Y).
        /// </summary>
        public double ContinuousMotionDutyFraction = 0.0;
        /// <summary>
        /// Defines the axis to be moved during scanning (for continuous motion) or between views.
        /// </summary>
        public MoveAxisForScan AxisToMove = MoveAxisForScan.X;
        /// <summary>
        /// XY speed for motion between views.
        /// </summary>
        public double XYSpeed = 20000.0;
        /// <summary>
        /// Acceleration for XY motion.
        /// </summary>
        public double XYAcceleration = 30000.0;
        /// <summary>
        /// Z speed for motion between layers.
        /// </summary>
        public double ZSpeed = 10000.0;
        /// <summary>
        /// Acceleration for Z motion.
        /// </summary>
        public double ZAcceleration = 10000.0;
        /// <summary>
        /// Time to wait in ms after the stop command is issued on each axis.
        /// </summary>
        public uint SlowdownTimeMS = 50;
        /// <summary>
        /// Position tolerance for X,Y,Z motion.
        /// </summary>
        public double PositionTolerance = 2.0;

        public QuasiStaticAcqSettings() : base("") {}

        public QuasiStaticAcqSettings(string name) : base(name) { }

        internal static QuasiStaticAcqSettings Default
        {
            get
            {
                QuasiStaticAcqSettings c = SySal.Management.MachineSettings.GetSettings(typeof(QuasiStaticAcqSettings)) as QuasiStaticAcqSettings;
                if (c == null) c = new QuasiStaticAcqSettings();
                return c;
            }
        }

        public override object Clone()
        {
            QuasiStaticAcqSettings S = new QuasiStaticAcqSettings(Name);
            S.AxisToMove = AxisToMove;
            S.BaseThickness = BaseThickness;
            S.ClusterThreshold = ClusterThreshold;
            S.ContinuousMotionDutyFraction = ContinuousMotionDutyFraction;
            S.EmulsionThickness = EmulsionThickness;
            S.FocusSweep = FocusSweep;
            S.Layers = Layers;
            S.MinValidLayers = MinValidLayers;
            S.Pitch = Pitch;
            S.PositionTolerance = PositionTolerance;
            S.Sides = Sides;
            S.SlowdownTimeMS = SlowdownTimeMS;
            S.ViewOverlap = ViewOverlap;
            S.XYAcceleration = XYAcceleration;
            S.XYSpeed = XYSpeed;
            S.ZAcceleration = ZAcceleration;
            S.ZSpeed = ZSpeed;            
            return S;
        }

        public void Copy(QuasiStaticAcqSettings c)
        {
            AxisToMove = c.AxisToMove;
            BaseThickness = c.BaseThickness;
            ClusterThreshold = c.ClusterThreshold;
            ContinuousMotionDutyFraction = c.ContinuousMotionDutyFraction;
            EmulsionThickness = c.EmulsionThickness;
            FocusSweep = c.FocusSweep;
            Layers = c.Layers;
            MinValidLayers = c.MinValidLayers;
            Pitch = c.Pitch;
            PositionTolerance = c.PositionTolerance;
            Sides = c.Sides;
            SlowdownTimeMS = c.SlowdownTimeMS;
            ViewOverlap = c.ViewOverlap;
            XYAcceleration = c.XYAcceleration;
            XYSpeed = c.XYSpeed;
            ZAcceleration = c.ZAcceleration;
            ZSpeed = c.ZSpeed;
        }

        public const string FileExtension = "quasistaticacq.config";

        internal static System.Xml.Serialization.XmlSerializer s_XmlSerializer = new System.Xml.Serialization.XmlSerializer(typeof(QuasiStaticAcqSettings));
    }
}
