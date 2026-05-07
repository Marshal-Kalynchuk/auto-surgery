use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    net::SocketAddr,
    sync::atomic::{AtomicBool, Ordering},
};

use bytemuck::{AnyBitPattern, CheckedBitPattern, NoUninit, Pod, Zeroable};
use flume::{Receiver, Sender};
use tokio::{net::UdpSocket, sync::Notify};

#[derive(Default)]
pub struct TriggeredBool {
    pub value: Cell<bool>,
    notify: Notify,
}

impl TriggeredBool {
    pub fn new() -> Self {
        Self {
            value: Cell::new(false),
            notify: Notify::new(),
        }
    }

    pub fn set(&self, value: bool) {
        self.value.set(value);
        self.notify.notify_waiters();
    }

    pub async fn wait_for_false(&self) {
        while self.value.get() {
            // info!("reevaluating...");
            self.notify.notified().await;
        }
    }

    pub async fn wait_on(&self) {
        while !self.value.get() {
            self.notify.notified().await;
        }
    }

    pub fn get(&self) -> bool {
        self.value.get()
    }
}

#[derive(Default)]
pub struct SharedTriggeredBool {
    pub value: AtomicBool,
    notify: Notify,
}

impl SharedTriggeredBool {
    pub fn new() -> Self {
        Self {
            value: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    pub fn set(&self, value: bool) {
        // self.value.set(value);
        self.value.store(value, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub async fn wait_for_false(&self) {
        while self.get() {
            // info!("reevaluating...");
            self.notify.notified().await;
        }
    }

    pub async fn wait_on(&self) {
        while !self.get() {
            self.notify.notified().await;
        }
    }

    pub fn get(&self) -> bool {
        self.value.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct SimpleChannel<T>(Sender<T>, Receiver<T>);

impl<T> Default for SimpleChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SimpleChannel<T> {
    pub fn new() -> Self {
        let (sender, recevier) = flume::unbounded();
        Self(sender, recevier)
    }
    pub fn send_blocking(&self, value: T) -> Result<(), flume::SendError<T>> {
        self.0.send(value)
    }
    pub async fn send(&self, value: T) -> Result<(), flume::SendError<T>> {
        self.0.send_async(value).await
    }
    pub fn try_recv(&self) -> Result<T, flume::TryRecvError> {
        self.1.try_recv()
    }
    pub async fn recv(&self) -> Result<T, flume::RecvError> {
        self.1.recv_async().await
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub const BYTES_PER_REMOTE_ARM: usize = 168;
pub const LR_NBYTES_TO_NAP: usize = BYTES_PER_REMOTE_ARM * 2;
pub const DOUBLES_PER_REMOTE_ARM: usize = 21;
pub const LR_N_DOUBLES_TO_NAP: usize = DOUBLES_PER_REMOTE_ARM * 2;

pub const DOUBLES_PER_ARM: usize = 38;
pub const BYTES_PER_ARM: usize = 8 * DOUBLES_PER_ARM;
pub const LR_NBYTES_FROM_NAP: usize = BYTES_PER_ARM * 2;
pub const LR_NDOUBLES_FROM_NAP: usize = DOUBLES_PER_ARM * 2;
pub const NAP_BUFFER_SIZE: usize = LR_NBYTES_FROM_NAP;
pub const VALID_FLAG: usize = 2012;

pub struct NapSdk {
    /// These are
    last_times: RefCell<[f64; 2]>,
    handcontroller_state: RefCell<[FromRobotToHCMessage; 2]>,
    from_plc_left: RefCell<FromHcToRobot>,
    from_plc_right: RefCell<FromHcToRobot>,
    // controlling_arm: Cell<Option<Arm>>,
    rm_id_in: Cell<f64>,
    rm_id_out: Cell<f64>,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Zeroable, NoUninit, CheckedBitPattern)]
pub enum Arm {
    Left,
    Right,
    Both,
}

impl Default for NapSdk {
    fn default() -> Self {
        Self::new()
    }
}

impl NapSdk {
    pub fn new() -> Self {
        Self {
            from_plc_left: FromHcToRobot::zeroed().into(),
            from_plc_right: FromHcToRobot::zeroed().into(),
            handcontroller_state: RefCell::new([FromRobotToHCMessage::zeroed(); 2]),
            last_times: RefCell::new([0.0, 0.0]),
            // controlling_arm: None.into(),
            rm_id_in: 0.0.into(),
            rm_id_out: 0.0.into(),
        }
    }
    // /// Checks which arm is currently controlling the robot.
    // fn get_controlling_arm(&self) -> Option<Arm> {
    //     self.controlling_arm.get()
    // }
    /// Updates the state of the given arm given some state.
    fn update_arm_state(&self, state: FromRobotToHCMessage, arm: Arm) {
        // println!("arm: {:?}, state: {:?}", arm, state);
        match arm {
            Arm::Left => self.handcontroller_state.borrow_mut()[0] = state,
            Arm::Right => self.handcontroller_state.borrow_mut()[1] = state,
            Arm::Both => {
                self.update_arm_state(state, Arm::Left);
                self.update_arm_state(state, Arm::Right);
            }
        }
    }
    fn unpack_controller_bytes(&self, data: &[u8]) -> bool {
        let left_hc_bytes =
            FromHcToRobot::from_raw_bytes(data[0..BYTES_PER_ARM].try_into().unwrap());
        let right_hc_bytes = FromHcToRobot::from_raw_bytes(
            data[BYTES_PER_ARM..LR_NBYTES_FROM_NAP].try_into().unwrap(),
        );

        // println!("Left HC bytes: {:?}", left_hc_bytes);

        if left_hc_bytes.valid == VALID_FLAG as f64
            && left_hc_bytes.datetime > self.last_times.borrow()[0]
        {
            *self.from_plc_left.borrow_mut() = left_hc_bytes;
            // println!("Left HC bytes: {:?}", left_hc_bytes);
            // if self.get_controlling_arm().is_none() {
            //     self.controlling_arm.set(Some(Arm::Left));
            // }
            self.last_times.borrow_mut()[0] = left_hc_bytes.datetime;
            return true;
        }
        if right_hc_bytes.valid == VALID_FLAG as f64
            && right_hc_bytes.datetime > self.last_times.borrow()[1]
        {
            *self.from_plc_right.borrow_mut() = right_hc_bytes;
            // if self.get_controlling_arm().is_none() {
            //     self.controlling_arm.set(Some(Arm::Right));
            // }
            self.last_times.borrow_mut()[1] = right_hc_bytes.datetime;
            return true;
        }
        false
    }
    /// Gets the current robot data for the controlling arm.
    fn get_robot_data(&self, arm: Arm) -> FromHcToRobot {
        match arm {
            Arm::Left => *self.from_plc_left.borrow(),
            Arm::Right => *self.from_plc_right.borrow(),
            Arm::Both => panic!("We cannot get data from both arms."),
        }
    }
    async fn poll_recv(&self, socket: &UdpSocket) -> std::io::Result<bool> {
        let mut buffer = [0u8; NAP_BUFFER_SIZE];
        let (size, _) = socket.recv_from(&mut buffer).await?;

        let buf = &buffer[..size];
        let to_unpack = &buf[buf.len() - LR_NBYTES_FROM_NAP..];

        // println!("ye");
        // println!("Received {} bytes from the hand controller", to_unpack.len());
        Ok(self.unpack_controller_bytes(to_unpack))
    }
    pub fn send_message_to_hand_controller(&self, msg: FromRobotToHCMessage, inner: Arm) {
        // if let Some(inner) = self.get_controlling_arm() {
        let new_rm_id = msg.valid_data;
        let rm_id = self.rm_id_in.get();
        if new_rm_id != rm_id {
            // Message has been updated.
            self.rm_id_in.set(new_rm_id);
            self.update_arm_state(msg, inner);
            // self.update_arm_state(msg, Arm::Right);
        }
        // }
    }
    async fn get_new_rm_id_out(&self) -> f64 {
        let handle = self.rm_id_out.get();
        let cycle_id = handle;
        self.rm_id_out.set(handle + 1.0);
        cycle_id
    }

    pub async fn poll(
        &self,
        socket: &UdpSocket,
        server_addr: SocketAddr,
    ) -> std::io::Result<Option<(FromHcToRobot, FromHcToRobot)>> {
        // Send out the frames, this needs to happen before sending.
        let frame = bytemuck::bytes_of(&*self.handcontroller_state.borrow()).to_vec();
        socket.send_to(&frame, &server_addr).await?;

        // Receive the response.
        Ok(if self.poll_recv(socket).await? {
            let mut l_pckt = self.get_robot_data(Arm::Left);
            l_pckt.cycle_id = self.get_new_rm_id_out().await;

            let mut r_pckt = self.get_robot_data(Arm::Right);
            r_pckt.cycle_id = self.get_new_rm_id_out().await;
            Some((l_pckt, r_pckt))
        } else {
            None
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FromRobotToHCMessage {
    /// Provides the absolute position on the X-axis.
    pub position: Vec3,
    /// Provides the absolute rotation on the X-axis. (radians)
    pub ori_x: Angle<Radians>,
    /// Provides the absolute rotation on the Y-axis. (radians)
    pub ori_y: Angle<Radians>,
    /// Provides the absolute rotation on the Z-axis. (radians)
    pub ori_z: Angle<Radians>,
    /// The following 6 angles are based on Craig's convention
    /// for DH parameters. For forward kinematics. See documentation
    /// for a description.
    ///
    /// These are in degrees.
    pub fk_angles: [Angle<Degrees>; 6],
    /// In radians.
    pub tool_roll_angle: Angle<Radians>,
    pub tool_type: f64,
    pub tool_actuation_override: f64,
    /// This is X, Y, Z.
    pub tool_force_vector: Vec3,
    pub robot_safety: BoolFloat,
    pub motion_enabled: BoolFloat,
    pub valid_data: f64,
}

pub const FROM_KUKA_LIST_SIZE: usize = 21;

pub trait MuckedFloats<const N: usize>: AnyBitPattern + NoUninit {
    fn from_raw_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), N * 8);
        bytemuck::pod_read_unaligned(&bytes[..N * 8])
    }
    fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl MuckedFloats<FROM_KUKA_LIST_SIZE> for FromRobotToHCMessage {}

impl FromRobotToHCMessage {
    pub fn as_float_array(&self) -> &[f64; FROM_KUKA_LIST_SIZE] {
        bytemuck::cast_ref(self)
    }
    pub fn as_mut_float_array(&mut self) -> &mut [f64; FROM_KUKA_LIST_SIZE] {
        bytemuck::cast_mut(self)
    }
}

/// A Robot message. This is directly cast back into an array,
/// so you must NOT change or add any fields as this will immediately
/// become invalid.
///
/// This direct cast is done for efficiency sake, but the named fields
/// add some degree of interpretability to this instead of just an opque
/// array of doubles.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FromHcToRobot {
    /// The change in the X-position.
    pub pos_x: f64,
    /// The change in the Y-position.
    pub pos_y: f64,
    /// The change in the Z-position.
    pub pos_z: f64,
    /// The change in the X-rotation in radians.
    pub rot_x: f64,
    /// The change in the Y-rotation in radians.
    pub rot_y: f64,
    /// The change in the Z-rotation in radians.
    pub rot_z: f64,
    /// The type of the tool that is being connected.
    /// There are three different tool types.
    pub tool_type: f64,
    pub tool_actuation_override: f64,
    pub safety_ws: f64,
    pub enable: f64,
    pub cycle_id: f64,
    pub shoulder_roll: f64,
    pub shoulder_yaw: f64,
    pub elbow: f64,
    /// Gimble angles.
    pub gimbal: [f64; 3],
    /// The force values Fx, Fy, Fz
    pub cartesian_forces: [f64; 3],
    /// X, Y, Z
    pub torque_absolute_reserved: [f64; 3],
    pub cartesian_positions: [f64; 3],
    /// X, Y, Z
    pub cartesian_rotations: [f64; 3],
    /// Tool velocity in meters per second
    pub tool_velocity: f64,
    pub reserved: [f64; 6],
    pub datetime: f64,
    pub valid: f64,
}

pub const ROBOT_PACKET_SIZE: usize = 38;

impl FromHcToRobot {
    pub fn from_raw_bytes(bytes: [u8; ROBOT_PACKET_SIZE * 8]) -> Self {
        bytemuck::pod_read_unaligned(&bytes)
    }
    // pub fn as_float_array(&self) -> &[f64; 38] {
    //     bytemuck::cast_ref(self)
    // }
    // pub fn as_mut_float_array(&mut self) -> &mut [f64; TO_KUKA_LIST_SIZE] {
    //     bytemuck::cast_mut(self)
    // }
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Radians;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Degrees;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Angle<T> {
    value: f64,
    _type: PhantomData<T>,
}

impl<T> Angle<T> {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            _type: PhantomData,
        }
    }
    pub fn set(&mut self, value: f64) {
        self.value = value;
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BoolFloat {
    value: f64,
}

impl BoolFloat {
    pub fn from_bool(value: bool) -> Self {
        Self {
            value: if value { 1.0 } else { 0.0 },
        }
    }
    pub fn get(&self) -> bool {
        self.value == 1.0
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[cfg(test)]
mod tests {
    use crate::sdk::{Angle, BoolFloat, FromRobotToHCMessage, Vec3};

    #[test]
    pub fn transmute_hm_message() {
        let message = FromRobotToHCMessage {
            position: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 2.0,
            },
            ori_x: Angle::new(3.0),
            ori_y: Angle::new(4.0),
            ori_z: Angle::new(5.0),
            fk_angles: [
                Angle::new(6.0),
                Angle::new(7.0),
                Angle::new(8.0),
                Angle::new(9.0),
                Angle::new(10.0),
                Angle::new(11.0),
            ],
            tool_roll_angle: Angle::new(12.0),
            tool_type: 13.0,
            tool_actuation_override: 14.0,
            tool_force_vector: Vec3 {
                x: 15.0,
                y: 16.0,
                z: 17.0,
            },
            robot_safety: BoolFloat::from_bool(true),
            motion_enabled: BoolFloat::from_bool(true),
            valid_data: 20.0,
        };

        assert_eq!(
            message.as_float_array(),
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 1.0, 1.0, 20.0
            ]
        );
    }
}
