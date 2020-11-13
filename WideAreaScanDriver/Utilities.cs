using System;
using System.Windows.Forms;
using SySal.OperaDb;

namespace SySal.DAQSystem.Drivers.WideAreaScanDriver
{
	/// <summary>
	/// Summary description for Utilities.
	/// </summary>
	public class Utilities 
	{
		public class ConfigItem
		{
			public string Name;
			public long Id;
			public ConfigItem(string newName, long newId)
			{
				Name = newName;
				Id = newId;
			}
			public override string ToString()
			{
				return Name.ToString ();
			}
		}		
		public Utilities()
		{
			//
			// TODO: Add constructor logic here
			//
		}
		public static void FillComboBox(ComboBox cmb, string query, OperaDbConnection conn)
		{
			cmb.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			try
			{				
				new OperaDbDataAdapter(query, conn).Fill(ds);
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
			}
			if (ds.Tables.Count != 0)
				foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				{
					cmb.Items.Add(new ConfigItem(dr[1].ToString(), System.Convert.ToInt64(dr[0])));
				}
		}

		public static void FillListBox(ListBox lbox, string query, OperaDbConnection conn)
		{
			lbox.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			try
			{				
				new OperaDbDataAdapter(query, conn).Fill(ds);
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message);
			}
			if (ds.Tables.Count != 0)
				foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				{
					lbox.Items.Add(new ConfigItem(dr[1].ToString(), System.Convert.ToInt64(dr[0])));
				}
		}

		public static void SelectId(ComboBox cmb, long id)
		{
			for (int i=0; i<cmb.Items.Count; i++)
			{
				if ( ((Utilities.ConfigItem)cmb.Items[i]).Id == id ) 
				{
					cmb.SelectedIndex = i;
					break;
				}
			}	
		}

		public static long WriteSettingsToDb(SySal.OperaDb.OperaDbConnection conn, string desc, string exe, int driverlevel, int marks, string settings)
		{		
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();						
			long authorid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM VW_USERS WHERE UPPER(USERNAME) = UPPER('" + cred.OPERAUserName + "') ", conn, null).ExecuteScalar());
			SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROGRAMSETTINGS(:description, :exe, :authorid, :driverlevel, :marks, :settings, :newid)", conn);					
			cmd.Parameters.Add("description", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = desc;
			cmd.Parameters.Add("exe", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input).Value = exe;
			cmd.Parameters.Add("authorid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = authorid;
			cmd.Parameters.Add("driverlevel", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = driverlevel;
			cmd.Parameters.Add("marks", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Input).Value = marks;
			cmd.Parameters.Add("settings", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input).Value = settings;
			cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);			
			try
			{
				cmd.ExecuteNonQuery();
				return (long) cmd.Parameters["newid"].Value;
				//	return 1;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message);
				return 0;
			}					
		}

		public static string LookupItem(long id, SySal.OperaDb.OperaDbConnection conn)
		{
			SySal.OperaDb.OperaDbCommand cmd = new OperaDbCommand("SELECT DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE ID=" + id, conn);
			object o =  cmd.ExecuteScalar();
			string name;
			if (o == null) name = id.ToString() + " <description not found>";
			else name = id.ToString() + " (" + o.ToString() +")";
			return name;
		}

	}
}
