use bytemuck::{Pod, Zeroable};
use nap_sdk::sdk::{Angle, BoolFloat, FromRobotToHCMessage, MuckedFloats, Radians, FromHcToRobot, Vec3, FROM_KUKA_LIST_SIZE};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::app::FROM_KUKA_BYTES_LEN;


pub const TO_KUKA_LIST_SIZE: usize = 11; //11;

/// A Kuka message. This is directly cast back into an array,
/// so you must NOT change or add any fields as this will immediately
/// become invalid.
/// 
/// This direct cast is done for efficiency sake, but the named fields
/// add some degree of interpretability to this instead of just an opque
/// array of doubles.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct KukaMessage {
    /// The change in position.
    pub position: Vec3,
    /// The change in the X-rotation in radians.
    pub rot_x: Angle<Radians>,
    /// The change in the Y-rotation in radians.
    pub rot_y: Angle<Radians>,
    /// The change in the Z-rotation in radians.
    pub rot_z: Angle<Radians>,
    /// The type of the tool that is being connected.
    /// There are three different tool types.
    pub tool_type: f64,
    pub tool_actuation_override: f64,
    pub safety_ws: f64,
    pub enable: BoolFloat,
    pub cycle_id: f64,
}

impl MuckedFloats<TO_KUKA_LIST_SIZE> for KukaMessage {}

impl KukaMessage {
    pub fn from_robot_message(robot: FromHcToRobot) -> Self {
        let bytes = robot.to_bytes();
        // println!("Bytes: {:?}", bytes);
        Self::from_raw_bytes(bytes[0..TO_KUKA_LIST_SIZE * 8].try_into().unwrap())
    }
    pub fn as_float_array(&self) -> &[f64; TO_KUKA_LIST_SIZE] {
        bytemuck::cast_ref(self)
    }
    pub fn as_mut_float_array(&mut self) -> &mut [f64; TO_KUKA_LIST_SIZE] {
        bytemuck::cast_mut(self)
    }
}


pub async fn read_kuka_message<R>(reader: &mut R) -> FromRobotToHCMessage
where 
    R: AsyncRead + Unpin
{
    let mut buffer = [0u8; FROM_KUKA_BYTES_LEN * 10];
    let size = reader.read_exact(&mut buffer).await.unwrap();
    let buffed = &buffer[..size];
    let last_message: [u8; FROM_KUKA_LIST_SIZE * 8] = buffed[buffed.len() - FROM_KUKA_BYTES_LEN..]
        .try_into()
        .unwrap();
    FromRobotToHCMessage::from_raw_bytes(&last_message)
}

#[cfg(test)]
mod tests {
    use crate::kuka::{Angle, BoolFloat, KukaMessage, Vec3};



    #[test]
    pub fn kuka_transmute() {

        let mut s = KukaMessage {
            position: Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            },
            rot_x: Angle::new(4.0),
            rot_y: Angle::new(5.0),
            rot_z: Angle::new(6.0),
            tool_type: 7.0,
            tool_actuation_override: 8.0,
            safety_ws: 9.0,
            enable: BoolFloat::from_bool(true),
            cycle_id: 11.0,
            
        };

        // let arr: [f64; 11] = bytemuck::cast_(s);
        assert_eq!(s.as_float_array(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 11.0]);

        s.position.x = 2.0;
        assert_eq!(s.as_float_array(), &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 11.0]);
        // println!("ARR: {:?}", arr);
    }
}