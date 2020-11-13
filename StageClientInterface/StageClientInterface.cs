using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.StageControl
{
    class StageHostInterface : MarshalByRefObject, SySal.StageControl.IStageWithoutTimesource, SySal.Management.IMachineSettingsEditor, SySal.Management.IManageable
    {
        public const string ConnectString = "tcp://127.0.0.1:1881/FlexStage.rem";

        public double CurrentTimeS
        {
            get { throw new Exception("Stub only."); }
        }

        #region IStageWithoutTimesource Members

        public void Dispose()
        {
            throw new Exception("Stub only.");
        }

        public void CancelRecording()
        {
            throw new Exception("Stub only.");
        }

        public double GetPos(Axis ax)
        {
            throw new Exception("Stub only.");
        }

        public AxisStatus GetStatus(Axis ax)
        {
            throw new Exception("Stub only.");
        }

        public ushort LightLevel
        {
            get
            {
                throw new Exception("Stub only.");
            }
            set
            {
                throw new Exception("Stub only.");
            }
        }

        public void PosMove(Axis ax, double pos, double speed, double acc, double dec)
        {
            throw new Exception("Stub only.");
        }

        public void Reset(Axis ax)
        {
            throw new Exception("Stub only.");
        }

        public void SpeedMove(Axis ax, double speed, double acc)
        {
            throw new Exception("Stub only.");
        }

        public void StartRecording(double mindeltams, double totaltimems)
        {
            throw new Exception("Stub only.");
        }

        public void Stop(Axis ax)
        {
            throw new Exception("Stub only.");
        }

        public TrajectorySample[] Trajectory
        {
            get { throw new Exception("Stub only."); ; }
        }

        #endregion

        public override object InitializeLifetimeService()
        {
            return null;
        }

        public StageHostInterface()
        {
            throw new Exception("Stub only.");
        }

        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            throw new Exception("Stub only.");
        }

        #endregion

        #region IManageable Members

        public SySal.Management.Configuration Config
        {
            get
            {
                throw new Exception("Stub only.");
            }
            set
            {
                throw new Exception("Stub only.");
            }
        }

        public SySal.Management.IConnectionList Connections
        {
            get { throw new Exception("Stub only."); ; }
        }

        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            throw new Exception("Stub only.");
        }

        public bool MonitorEnabled
        {
            get
            {
                throw new Exception("Stub only.");
            }
            set
            {
                throw new Exception("Stub only.");
            }
        }

        public string Name
        {
            get
            {
                throw new Exception("Stub only.");
            }
            set
            {
                throw new Exception("Stub only.");
            }
        }

        #endregion
    }

    public class StageClientInterface : IStage, SySal.Management.IManageable, SySal.Management.IMachineSettingsEditor
    {
        StageHostInterface m_HI;
        System.Diagnostics.Stopwatch m_Watch = new System.Diagnostics.Stopwatch();
        double TrajectoryTimeDeltaMS = 0.0;
        System.Diagnostics.Process m_HP;

        public StageClientInterface()
        {
            string file = System.Reflection.Assembly.GetExecutingAssembly().Location;
            file = file.Substring(0, file.LastIndexOfAny(new char[] { '/', '\\' }) + 1) + "StageHostProcess.exe";
            try
            {
                System.Diagnostics.ProcessStartInfo psi = new System.Diagnostics.ProcessStartInfo(file);
                m_HP = new System.Diagnostics.Process();
                m_HP.StartInfo = psi;
                m_HP.Start();
                System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new System.Runtime.Remoting.Channels.Tcp.TcpChannel(), false);
                m_HI = (StageHostInterface)System.Runtime.Remoting.RemotingServices.Connect(typeof(StageHostInterface), SySal.StageControl.StageHostInterface.ConnectString) as StageHostInterface;
            }
            catch (Exception x)
            {
                throw x;
            }
        }

        ~StageClientInterface()
        {
            if (m_HP != null) Dispose();
        }

        #region IStage Members

        public System.Diagnostics.Stopwatch TimeSource
        {
            set { m_Watch = value; }
        }

        #endregion

        #region IStageWithoutTimesource Members

        public void CancelRecording()
        {
            m_HI.CancelRecording();
        }

        public double GetPos(Axis ax)
        {
            return m_HI.GetPos(ax);
        }

        public AxisStatus GetStatus(Axis ax)
        {
            return m_HI.GetStatus(ax);
        }

        public ushort LightLevel
        {
            get
            {
                return m_HI.LightLevel;
            }
            set
            {
                m_HI.LightLevel = value;
            }
        }

        public void PosMove(Axis ax, double pos, double speed, double acc, double dec)
        {
            m_HI.PosMove(ax, pos, speed, acc, dec);
        }

        public void Reset(Axis ax)
        {
            m_HI.Reset(ax);
        }

        public void SpeedMove(Axis ax, double speed, double acc)
        {
            m_HI.SpeedMove(ax, speed, acc);
        }

        public void StartRecording(double mindeltams, double totaltimems)
        {
            double e1, e2, e3;
            do
            {
                e1 = m_Watch.Elapsed.TotalSeconds;
                e2 = m_HI.CurrentTimeS;
                e3 = m_Watch.Elapsed.TotalSeconds;
            }
            while (e3 - e1 > 0.001);
            m_HI.StartRecording(mindeltams, totaltimems);
            TrajectoryTimeDeltaMS = 1000.0 * (0.5 * (e1 + e3) - e2);
        }

        public void Stop(Axis ax)
        {
            m_HI.Stop(ax);
        }

        public TrajectorySample[] Trajectory
        {
            get
            {
                TrajectorySample[] s = m_HI.Trajectory;
                int i;
                for (i = 0; i < s.Length; i++)
                    s[i].TimeMS += TrajectoryTimeDeltaMS;
                return s;
            }
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            if (m_HP != null)
            {
                m_HP.CloseMainWindow();
                m_HP = null;
            }
            GC.SuppressFinalize(this);
        }

        #endregion

        #region IManageable Members

        public SySal.Management.Configuration Config
        {
            get
            {
                return m_HI.Config;
            }
            set
            {
                m_HI.Config = value;
            }
        }

        public SySal.Management.IConnectionList Connections
        {
            get { return m_HI.Connections; }
        }

        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            return m_HI.EditConfiguration(ref c);
        }

        public bool MonitorEnabled
        {
            get
            {
                return m_HI.MonitorEnabled;
            }
            set
            {
                m_HI.MonitorEnabled = value;
            }
        }

        public string Name
        {
            get
            {
                return m_HI.Name;
            }
            set
            {
                m_HI.Name = value;
            }
        }

        #endregion

        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            return m_HI.EditMachineSettings(t);
        }

        #endregion
    }
}
