using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.StageControl
{
    public class StageHostInterface : MarshalByRefObject, SySal.StageControl.IStageWithoutTimesource, SySal.Management.IMachineSettingsEditor, SySal.Management.IManageable 
    {
        public const string ConnectString = "tcp://127.0.0.1:1881/FlexStage.rem";

        FlexStage m_Stage;
        System.Diagnostics.Stopwatch m_HostWatch = System.Diagnostics.Stopwatch.StartNew();        

        public double CurrentTimeS 
        {
            get { return m_HostWatch.Elapsed.TotalSeconds; }
        }

        #region IStageWithoutTimesource Members

        public void Dispose()
        {
            if (m_Stage != null)
            {
                m_Stage.Dispose();
                m_Stage = null;
                GC.SuppressFinalize(this);
            }
        }

        ~StageHostInterface()
        {
            if (m_Stage != null) Dispose();
        }

        public void CancelRecording()
        {
            m_Stage.CancelRecording();
        }

        public double GetPos(Axis ax)
        {
            return m_Stage.GetPos(ax);
        }

        public AxisStatus GetStatus(Axis ax)
        {
            return m_Stage.GetStatus(ax);
        }

        public ushort LightLevel
        {
            get
            {
                return m_Stage.LightLevel;
            }
            set
            {
                m_Stage.LightLevel = value;
            }
        }

        public void PosMove(Axis ax, double pos, double speed, double acc, double dec)
        {
            m_Stage.PosMove(ax, pos, speed, acc, dec);
        }

        public void Reset(Axis ax)
        {
            m_Stage.Reset(ax);
        }

        public void SpeedMove(Axis ax, double speed, double acc)
        {
            m_Stage.SpeedMove(ax, speed, acc);
        }

        public void StartRecording(double mindeltams, double totaltimems)
        {
            m_Stage.StartRecording(mindeltams, totaltimems);
        }

        public void Stop(Axis ax)
        {
            m_Stage.Stop(ax);
        }

        public TrajectorySample[] Trajectory
        {
            get { return m_Stage.Trajectory; }
        }

        #endregion

        public override object InitializeLifetimeService()
        {
            return null;
        }

        public StageHostInterface()
        {
            m_Stage = new FlexStage();
            m_Stage.TimeSource = m_HostWatch;            
        }

        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            return m_Stage.EditMachineSettings(t);
        }

        #endregion

        #region IManageable Members

        public SySal.Management.Configuration Config
        {
            get
            {
                return m_Stage.Config;
            }
            set
            {
                m_Stage.Config = value;
            }
        }

        public SySal.Management.IConnectionList Connections
        {
            get { return m_Stage.Connections; }
        }

        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            SySal.Management.Configuration C = c;
            bool ret = m_Stage.EditConfiguration(ref C);
            c = C;
            return ret;
        }

        public bool MonitorEnabled
        {
            get
            {
                return m_Stage.MonitorEnabled;
            }
            set
            {
                m_Stage.MonitorEnabled = value;
            }
        }

        public string Name
        {
            get
            {
                return m_Stage.Name;
            }
            set
            {
                m_Stage.Name = value;
            }
        }

        #endregion
    }
}